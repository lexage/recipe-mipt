# SWE-rebench integration contract

This document fixes the boundary between prediction generation in this repository
and evaluation in
[`SWE-rebench/SWE-bench-fork`](https://github.com/SWE-rebench/SWE-bench-fork).
It is the source of truth for the first implementation of the SWE-rebench runner.

## Versioning policy

SWE-rebench and its evaluation fork can change independently of this repository.
Every experiment must therefore record the following values in its run metadata:

- dataset name or local dataset path;
- dataset split and, when available, dataset revision;
- commit SHA of `SWE-rebench/SWE-bench-fork`;
- prediction generator commit SHA;
- pipeline configuration and model name;
- evaluation `run_id`.

The initial defaults for the integration are:

| Setting | Default |
| --- | --- |
| Dataset | Explicit CLI value, for example `nebius/SWE-rebench` |
| Split | Explicit CLI value |
| Evaluator | a separately checked out `SWE-rebench/SWE-bench-fork` |
| Prediction serialization | JSONL, one object per line |

Both the dataset name and split must remain CLI options. A local `.json` or
`.jsonl` dataset path must also be accepted, because the SWE-bench harness loader
supports local files and this makes runs reproducible after pinning a snapshot.

For Hub datasets, create that snapshot before inference and pass the resulting
path to both generation and evaluation. The snapshot intentionally retains the
evaluator-only fields, while the inference loader strips them before building a
task or prompt:

```bash
python snapshot_swe_rebench.py \
  --dataset nebius/SWE-rebench \
  --split test \
  --dataset-revision <immutable-hugging-face-commit> \
  --instance-ids-file instance_ids.txt \
  --output /absolute/path/to/pinned-swe-rebench.jsonl
```

The command also writes `pinned-swe-rebench.jsonl.metadata.json` containing the
source revision, selected IDs, and SHA-256 digest. A remote dataset without an
explicit immutable revision is rejected.

## Dataset record

Prediction generation consumes these fields from every record:

| Field | Required by the generator | Meaning |
| --- | --- | --- |
| `instance_id` | yes | Stable identifier used to join a prediction to an evaluation instance. |
| `repo` | yes | GitHub repository in `owner/name` form. |
| `base_commit` | yes | Commit on which the agent workspace must be based. |
| `problem_statement` | yes | Issue presented to the agent. |
| `hints_text` | no | Optional hints that may be included according to runner policy. |
| `version` | no | Repository version used by the evaluation environment. |

The dataset may contain evaluator-only fields such as `patch`, `test_patch`,
`FAIL_TO_PASS`, and `PASS_TO_PASS`. They must never be included in the agent
prompt or exposed through repository tools: doing so would leak the reference
solution or hidden evaluation tests.

The loader must fail before inference when a required field is absent, empty, or
has the wrong type. It must also reject duplicate `instance_id` values.

## Prediction record

The output file passed to the harness contains one JSON object per line with
exactly these fields:

```json
{
  "instance_id": "owner__repo-123",
  "model_name_or_path": "react-sgr/model-name",
  "model_patch": "diff --git a/src/file.py b/src/file.py\n..."
}
```

Field requirements:

- `instance_id` must exist in the selected dataset split and occur at most once;
- `model_name_or_path` identifies the pipeline/model combination and must be a
  non-empty string;
- `model_patch` is the repository diff against `base_commit` and must be a
  string; the runner writes an empty patch for an empty solution, timeout, or
  inference failure so every selected instance remains in the score denominator.

Generation logs, final natural-language answers, tool traces, timings, and error
details do not belong in this file. They must be written to separate metadata,
log, and error files so the evaluator input remains strict and portable.

## Evaluation command

Evaluation is a separate stage and is run from the checked-out fork:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name nebius/SWE-rebench \
  --split test \
  --predictions_path /absolute/path/to/predictions.jsonl \
  --max_workers 4 \
  --run_id <unique-run-id>
```

Use `evaluate_swe_rebench.py` for the actual hand-off; it additionally requires
the matching inference `run_metadata.json`, an explicit namespace, and verifies
that dataset, revision, split, and selected IDs match before invoking the fork.

For a smoke run, append the selected identifiers:

```bash
  --instance_ids <instance-id-1> <instance-id-2>
```

The evaluator requires a working Docker installation for local execution. The
number of workers, timeout, image namespace, cache policy, and other harness
options belong to the evaluation wrapper rather than the prediction generator.

## Pipeline configurations

Ready-to-customize configurations for both target pipelines are provided in
`pipeline_configs/react_sgr_swe_rebench.yaml` and
`pipeline_configs/rewoo_sgr_swe_rebench.yaml`. Both expose only repository-bound
tools during inference. Change their model endpoint and model name for the
deployment, then pass the selected file to `run_swe_rebench.py --config`.

Before a full run, the wrapper must execute
`python -m swebench.harness.run_evaluation --help` against the pinned fork and
fail with an actionable message if the command-line contract above is not
supported by that revision.

The repository provides that wrapper as `evaluate_swe_rebench.py`. Validate the
handoff without starting evaluation containers first:

```bash
python evaluate_swe_rebench.py \
  --fork-path /path/to/SWE-bench-fork \
  --predictions-path /absolute/path/to/predictions.jsonl \
  --run-id react-sgr-smoke \
  --instance-ids <instance-id> \
  --check-only
```

Remove `--check-only` to start the official evaluator. The wrapper validates the
strict JSONL schema, rejects duplicate predictions, checks the fork's CLI, and
then launches the harness in a subprocess. It never imports evaluator internals.
It also writes `evaluation_metadata.json` into `--report-dir`, including the
fork and generator revisions and the exact harness command.

The pinned evaluator currently writes its aggregate report in the fork checkout
and instance reports under `logs/run_evaluation`. After a successful run, the
wrapper copies both into `--report-dir` and fails if no report can be found.
An empty namespace is passed to the fork verbatim so that `--namespace ""`
continues to select locally built images.

## One-instance smoke run

Before allocating workers for a full run, execute the complete generation to
evaluation hand-off for one instance:

```bash
python smoke_swe_rebench.py \
  --config pipeline_configs/react_sgr_swe_rebench.yaml \
  --dataset /absolute/path/to/pinned-swe-rebench.jsonl \
  --split test \
  --instance-id <instance-id> \
  --fork-path /path/to/SWE-bench-fork \
  --image-manifest /path/to/pinned-images.json \
  --output-dir runs/react-sgr-smoke \
  --model-name-or-path react-sgr/<model> \
  --run-id react-sgr-smoke \
  --namespace swerebench
```

The smoke runner forces one inference worker and one evaluation worker, writes
the selected ID and predictions under `--output-dir`, and stops immediately if
inference fails. Add `--evaluation-check-only` to validate the hand-off without
starting the official evaluation container.

## Stage boundary

Prediction generation owns:

1. loading and filtering instances;
2. creating an isolated checkout at `base_commit`;
3. running `react-sgr` or `rewoo-sgr` with repository tools;
4. collecting `git diff --binary <base_commit>`;
5. validating and writing prediction records;
6. recording generation errors without aborting unrelated instances.

The external fork owns:

1. applying `model_patch` in the official environment;
2. building or retrieving evaluation images;
3. running the benchmark tests;
4. determining whether an instance is resolved;
5. producing evaluation logs and aggregate reports.

The generation process must not import private evaluator internals. Its only
integration artifact is the prediction JSONL file.

## Container isolation invariant

Real benchmark inference runs in a dedicated container created from the
instance image for that task. Repository tools operate only on `/testbed` in
that inference container, and the container is stopped and removed after its
`model_patch` has been collected.

Evaluation must never reuse an inference container. The external fork creates a
fresh container from the clean instance image, applies only `model_patch`, and
runs the official evaluation scripts there. This prevents packages, caches,
environment changes, or files created during inference but absent from the
patch from affecting the reported score.

Inference failures are written to `errors.jsonl` with the original remote error
type, traceback, duration, and bounded diagnostics. For an empty patch the
diagnostics include the pipeline's final result, task image, Git status, and diff
summary so a model termination can be distinguished from a Docker or Git error.

The inference CLI configures the existing agent logging statements through the
shared `create_logging` helper. `--logs-path` overrides the pipeline YAML path;
the per-run log also records tab-separated `STAGE` events for dataset loading,
image resolution, worker and pipeline lifecycle, Docker startup/validation,
patch collection, timeouts, cleanup, and the final run summary.

## Pinned inference images

Production inference first uses the exact `image_name` or `docker_image` stored
in the pinned dataset record. A manifest remains supported for replaying an
older run; deriving image names from an instance ID is disabled by default. A
manifest must come from the same pinned evaluator revision and is a JSON object
keyed by every selected instance ID:

```json
{
  "owner__repo-123": {
    "instance_image_key": "registry/exact-image@sha256:...",
    "platform": "linux/amd64",
    "workdir": "/testbed",
    "user": "root",
    "cap_add": []
  }
}
```

The inference CLI requires explicit `--dataset` and `--split` values. The
evaluation CLI additionally requires an explicit namespace; pass `swerebench`
for leaderboard images or an empty value for locally built images, following
the pinned fork's documentation.

An absent instance is a hard inference error. Convention-based image names are
available only through the explicit `--allow-image-convention` escape hatch for
local smoke tests and must not be used for reported benchmark runs.

## Compatibility gate

The integration is ready for a full benchmark run only after all of these checks
pass for at least one real instance:

1. the pinned dataset loads with the selected split;
2. required dataset fields pass local validation;
3. the generated patch applies to a clean checkout of `base_commit`;
4. the prediction JSONL passes local schema validation;
5. the pinned fork accepts the file and starts evaluation;
6. dataset revision, fork SHA, generator SHA, model, pipeline config, and
   `run_id` are present in run metadata.

If the upstream dataset identifier or CLI differs in the pinned revision, update
this contract and the integration together before running predictions.

An opt-in real integration check is available for the final compatibility gate:

```bash
SWE_REBENCH_REAL_TASK=/absolute/path/to/one-pinned-task.jsonl \
SWE_REBENCH_EVALUATOR_FORK=/absolute/path/to/SWE-bench-fork \
pytest -q tests/integration/test_swe_rebench_real.py
```

It starts and removes two independent containers for the same task, validates
the exact checkout commit, and runs the evaluator compatibility check against
the real pinned fork.
