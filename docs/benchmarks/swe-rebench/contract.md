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
| Dataset | `SWE-rebench/SWE-rebench` |
| Split | `test` |
| Evaluator | a separately checked out `SWE-rebench/SWE-bench-fork` |
| Prediction serialization | JSONL, one object per line |

Both the dataset name and split must remain CLI options. A local `.json` or
`.jsonl` dataset path must also be accepted, because the SWE-bench harness loader
supports local files and this makes runs reproducible after pinning a snapshot.

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
  string; an empty patch is a valid harness input but represents no proposed
  change and should be reported as an unsuccessful generation by the runner.

Generation logs, final natural-language answers, tool traces, timings, and error
details do not belong in this file. They must be written to separate metadata,
log, and error files so the evaluator input remains strict and portable.

## Evaluation command

Evaluation is a separate stage and is run from the checked-out fork:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-rebench/SWE-rebench \
  --split test \
  --predictions_path /absolute/path/to/predictions.jsonl \
  --max_workers 4 \
  --run_id <unique-run-id>
```

For a smoke run, append the selected identifiers:

```bash
  --instance_ids <instance-id-1> <instance-id-2>
```

The evaluator requires a working Docker installation for local execution. The
number of workers, timeout, image namespace, cache policy, and other harness
options belong to the evaluation wrapper rather than the prediction generator.

Before a full run, the wrapper must execute
`python -m swebench.harness.run_evaluation --help` against the pinned fork and
fail with an actionable message if the command-line contract above is not
supported by that revision.

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
