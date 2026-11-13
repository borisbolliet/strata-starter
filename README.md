# strata-starter

Starter code for the C1 Research Computing and Software Development coursework,
University of Cambridge, Michaelmas 2026. The coursework brief gives the questions
and formulas.

## Installation

Use Python 3.12 with the supplied dependency files. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m pip install -r requirements-submission.txt
python -m pip install -e . --no-deps
python -c "import jax; print(jax.devices())"
```

On the course TPU VM, use `requirements-tpu.lock` in place of
`requirements-cpu.lock`. Python 3.12 must be installed before these commands.

## Supplied code

| Module or function | Purpose |
|---|---|
| `strata.sim.cohort(seed, n, p)` | Generate the cohort; `geno` contains the genotype matrix and `pheno` the trait vector. |
| `strata.reference` | NumPy reference implementations, using float64 by default. |
| `strata.angles.principal_angles(Q1, Q2)` | Compare two subspaces; returns angles in degrees, in ascending order. |
| `strata.bench.timeit(f, *args)` | Time execution after the first call and wait for each computation to finish. |

Write the five JAX functions in `src/strata/kernels.py` and add the tests requested
in Question 1. The supplied NumPy implementations are references for those tests.

## Submitting answers

Submit your work to the **private GitLab project assigned by the course**.
This public repository is for distributing the starter. Do not submit answers
through its issues or pull requests.

Each student submits `submissions/<crsid>/answers.json`. Replace `<crsid>` with
your own CRSid. For example:

```bash
mkdir -p submissions/bb667
cp answers.template.json submissions/bb667/answers.json
```

Replace every `null` with an answer. Keep the keys and case identifiers unchanged.
The file contains numeric results, short explanations and paths to your code,
test log and plots. It replaces the PDF report and the separate `predictions.json`.

| Question | JSON section | Contents |
|---|---|---|
| 1(a) | `q1.a` | Path to the JAX implementation. |
| 1(b) | `q1.b` | Test files, execution log and measured errors. |
| 2(a) | `q2.a` | Eleven coefficients and a one-sentence justification for each. |
| 2(b) | `q2.b` | Predictions for the eleven specified kernel/size combinations. |
| 2(c-e) | `q2.c`, `q2.d`, `q2.e` | Scaling exponents, batch-size threshold and memory-capacity limit. |
| 3(a) | `q3.a` | Three GRM measurements, buffer sizes, scaling comparison and host-CPU comparison. |
| 3(b) | `q3.b` | Eigenvector timings, their ratio and the two angles. |
| 3(c) | `q3.c` | Six scan measurements, the threshold comparison and the scan plot. |
| 4(a) | `q4.a` | Path to the roofline plot. |
| 4(b) | `q4.b` | Your interpretation, at most 150 words. |

`q3.environment` records the TPU device descriptions and JAX version. Questions
1 and 2 are individual work. Team members may share Question 3 measurements and
plots; each student writes their own explanations and Question 4(b) interpretation.

### Units and files

- Times are in **seconds**, rates in **FLOP/s**, and memory sizes in **bytes**.
- `percent_peak` is a percentage; `roof_time_over_time` and `speedup` are ratios.
- Enter JSON numbers, not strings containing units. `1.2e-3` is valid JSON.
- Store plots as PDF or PNG. Paths are relative to the project root, using `/`.
  Commit every referenced file, including the test log.
- The two Question 1 angles come from the full `n × k` outputs. Question 3(b)
  compares the first two columns, as specified in the brief.

The schema in `answers.schema.json` describes each field. To save results from
Python, use `json.dump(answers, file, indent=2, allow_nan=False)`; convert NumPy or
JAX scalars with `float(...)` or `int(...)`, and arrays with `.tolist()`.

### Validation

Run from the project root:

```bash
python validate_submission.py submissions/bb667/answers.json
```

The validator checks required fields, data types, finite numbers, file paths and
the 150-word limit. It does not mark the answers or execute the submitted code.
A successful check also requires the referenced files to exist and be non-empty.

During preparation, `--draft` permits unanswered `null` fields and skips checking
whether the referenced files exist:

```bash
python validate_submission.py submissions/bb667/answers.json --draft
```

Run the full check before submitting. Use tags `<crsid>-q1` through `<crsid>-q4`
to identify your commits for each question without colliding with teammates' tags.
