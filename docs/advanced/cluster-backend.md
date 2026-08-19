---
title: Ray and Kubernetes Backend
description: Run the same training script on Ray, or let Kubernetes schedule every worker of the run.
---
Miles runs a job on one of two cluster backends. Ray is the default and needs no configuration.
Kubernetes installs the run as a helm release, so the cluster schedules every worker. The training
script is the same either way.

<Warning>

**Status.** Under active development: flags, chart values and failure semantics still change. Ray
also runs *inside* Kubernetes, and that is the well-trodden path — take this one only if you want
Miles to create the cluster's objects itself.

</Warning>

## Launch

Run everything from the repository root, with `kubectl` and `helm` on your PATH.

Install a workbench — the long-lived pod you launch from — once per namespace, from your
cluster's [`infra.yaml`](#for-cluster-administrator):

```bash
export MILES_NS="miles-$USER"
python -m miles.utils.external_utils.miles_workbench install -n "$MILES_NS" -f infra.yaml
```

It creates the namespace if missing, checks your rights, installs, and waits for Ready. Runs
launched from the pod inherit those values.

Launch from inside it:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && python scripts/run_qwen3_4b.py train"
```

## Observability

**Built in**

- The launcher follows every pod's logs, prints status changes and warning events, and ends with
  the run's exit code.
- On a failure, collect the namespace's logs, describes and events into one directory — before
  cleaning up, and the whole namespace, because the explanation is usually the pod next to it:
  `python -m miles.utils.external_utils.miles_workbench collect-diagnosis -n "$MILES_NS" --output-dir ~/artifacts/miles`

**External**

- A run's pods are ordinary pods, so whatever the cluster already runs — a metrics stack, a log
  collector, the platform's own dashboards — sees them with no wiring from Miles.
- Prefer it at scale. The built-in following is meant for watching one run, not hundreds of pods.

## Clean up

```bash
python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" 260811-143000-042
python -m miles.utils.external_utils.miles_workbench uninstall -n "$MILES_NS"
```

`stop` removes the run and frees its GPUs; `uninstall` removes the workbench.

## Colocated engines

Under `--colocate` the trainer asks for whole nodes and the engines ask for no GPU at all: a
scheduling gate holds each engine pod until the trainer it shares with has landed, and a pairing
controller then pins it to that node.

Engines narrower than a node are fine — eight `--rollout-num-gpus-per-engine 1` engines share one
8-GPU trainer node. Such an engine sees every card on the node, so it is told at startup which one
is its own, computed from the pod's own position rather than rendered into the command.

Each pool pod gets a memory-backed `/dev/shm`, the same shared memory the docker quick start asks
for with `--shm-size`. Kubernetes' own default is 64Mi, and NCCL wants tens of Mi for every peer it
cannot reach over p2p, so a pool holding part of a node's cards would otherwise die at rendezvous
with `No space left on device`.

Three limits remain:

- The trainer still takes whole nodes. A trainer pod narrower than a node is refused.
- An engine cannot be wider than a node, since nothing would place the half of it that reaches
  past the node it is pinned to.
- Losing one of several narrow engines ends the run. The pairing controller does put the replacement
  pod back on the trainer's node, but the weight update that runs while it is away waits on a rank
  that is not coming, and a trainer of a single cell has no second cell to retry from. A run of one
  whole-node engine does not hit this: losing that engine leaves none to broadcast to at all.

## Folder convention

A run is many pods on many machines, and they share nothing but the storage `infra.yaml` mounts.
A path that is not on it is the most common way a run fails.

- Every path your script names — `/root/models`, `/root/datasets` — has to be on it.
- Copying a file into a pod is pointless: pods come and go, the mount survives.
- A `hostPath` has to hold on every node a pod can land on. Where only part of the cluster mounts
  the share, say so in `infra.scheduling.nodeSelector`: a node that has the directory but not the
  filesystem under it takes the writes onto its own disk, and the run reads an empty share back.
- To run your own branch instead of the image's copy, name its sub-path under the storage root:
  `infra.paths.repos.miles: alice/miles`. `megatron` and `sglang` work the same way.

## For cluster administrator

Everything above assumes this was done once.

**Install LWS.** Miles deploys its worker pools as
[LeaderWorkerSets](https://github.com/kubernetes-sigs/lws). Install the CRDs and controller, and
grant users rights over them explicitly: LWS ships no aggregation labels, so a namespace `admin`
role does not include them.

**Give each user a namespace.** The namespace is the real boundary, not the Role: anything that
may create workloads can name another ServiceAccount and read its token. Keep privileged accounts
out of it.

**Write one `infra.yaml`.** The same file drives every Miles chart:

```yaml
infra:
  image:
    repository: radixark/miles
    tag: dev
  sharedStorage:
    type: hostPath
    hostPath: /cluster-storage
    mountPath: /cluster-storage
```

`charts/miles-run/values.yaml` shows the full shape, and each chart's `values.schema.json` is the
authoritative field list.
