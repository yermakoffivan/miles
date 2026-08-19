import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import SOLVER_MODEL_ID, VERIFIER_MODEL_ID, ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_multi_policy import TrainRewardBounds, execute

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=7000,
    suite="stage-c-8-gpu-h100",
    labels=["long", "multi-policy", "fully-async"],
    disabled=(
        "the run stops producing rollouts around solver=99 verifier=80 and never resumes, then dies on the "
        "harness clock. It stalls 17 minutes into a 117 minute budget, so this is the end of the rollout count "
        "and not the end of the time: raising est_time did not help, and lowering NUM_ROLLOUT would only starve "
        "the trailing policy sooner. What is lost meanwhile is real and the short variant does not replace it: "
        "this is the only multi-policy soak, and its solver bound of final_min=0.5 is the only assertion "
        "anywhere that a multi-policy run learns at all -- the short variant's final_min=0.01 only proves the "
        "reward fired once. Re-enable by fixing the stall, not by loosening these bounds."
    ),
)

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "100"))

# TODO: tighten these weak bounds once the e2e run has been observed.
TRAIN_REWARD_BOUNDS = {
    SOLVER_MODEL_ID: TrainRewardBounds(initial_max=0.6, final_min=0.5),
    VERIFIER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.1),
}


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        args,
        wandb_args=command_utils.get_default_wandb_args(__file__),
        train_reward_bounds=TRAIN_REWARD_BOUNDS,
    )
