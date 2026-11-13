# strata-starter

Starter code for the C1 Research Computing and Software Development coursework,
University of Cambridge, Michaelmas 2026. The coursework brief gives the questions
and formulas.

## Installation

Use Python 3.12 with the supplied dependency files. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -c "import jax; print(jax.devices())"
```

On the course TPU VM, install `requirements-tpu.lock` and
`requirements-submission.txt` in place of `requirements.txt`. Python 3.12 must
be installed before these commands.

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

Each student submits:

- `report/<crsid>/report.pdf`: an individual report of at most **3000 words**,
  organised by Questions 1–4 and their parts, with explanations, comparisons
  and labelled plots.
- `results/<crsid>/results.json`: the numerical results and paths to the report,
  code, test log and plots, using the supplied format.

The CRSid subdirectories keep the three students' submissions separate within
one team repository. The supplied `report/report.tex` and `results/results.json`
are templates, not completed submissions.

The word limit is an upper limit; a shorter report can receive full marks.
Question 4(b) retains its limit of 150 words within the report. Use selected tables
where they help the discussion; the JSON records the complete numerical results.
The two plots must appear in the report and also be saved as separate files.

Replace `<crsid>` with your own CRSid. For example:

```bash
mkdir -p report/bb667 results/bb667
cp results/results.json results/bb667/results.json
cp report/report.tex report/bb667/report.tex
```

The LaTeX report template is optional; any editor that exports PDF is suitable.
For LaTeX, edit the template and compile it from the project root:

```bash
pdflatex -output-directory=report/bb667 report/bb667/report.tex
```

Replace every `null` in the JSON and keep its keys and case identifiers unchanged.
Set `report_path` to `report/bb667/report.pdf`, using your own CRSid.

| Question | JSON section | Results to record |
|---|---|---|
| 1(a) | `q1.a` | Path to the JAX implementation. |
| 1(b) | `q1.b` | Test files, execution log and measured errors. |
| 2(a) | `q2.a` | Eleven coefficient values; explain the counts in the report. |
| 2(b) | `q2.b` | Predictions for the eleven specified kernel/size combinations. |
| 2(c-e) | `q2.c`, `q2.d`, `q2.e` | Scaling exponents, batch-size threshold and memory-capacity limit. |
| 3(a) | `q3.a` | Three GRM measurements, buffer sizes, fitted exponent and host-CPU comparison. |
| 3(b) | `q3.b` | Eigenvector timings, their ratio and the two angles. |
| 3(c) | `q3.c` | Six scan measurements, the first tested crossing and the scan plot. |
| 4(a) | `q4.a` | Path to the roofline plot. |
| 4(b) | Report only | Interpretation, at most 150 words. |

`q3.environment` records the TPU device descriptions and JAX version. Questions
1 and 2 are individual work. Team members may share Question 3 measurements and
plots; each student writes their own report, including all explanations and the
Question 4(b) interpretation.

Use the supplied JSON template, format version 2.0.

### Units and files

- Times are in **seconds**, rates in **FLOP/s**, and memory sizes in **bytes**.
- `percent_peak` is a percentage; `roof_time_over_time` and `speedup` are ratios.
- Enter JSON numbers, not strings containing units. `1.2e-3` is valid JSON.
- Store plots as PDF or PNG. Paths are relative to the project root, using `/`.
  Commit every referenced file, including the test log.
- The two Question 1 angles come from the full `n × k` outputs. Question 3(b)
  compares the first two columns, as specified in the brief.

The schema in `results.schema.json` describes each field. To save results from
Python, use `json.dump(results, file, indent=2, allow_nan=False)`; convert NumPy or
JAX scalars with `float(...)` or `int(...)`, and arrays with `.tolist()`.

### Validation

Run from the project root to check all completed submissions in the team repo:

```bash
python check_submission.py
```

To check only your submission:

```bash
python check_submission.py results/bb667/results.json
```

The checker follows the same approach as the S1 `check_submission.py`: it checks
the README, opens the report PDF, validates the JSON format and checks referenced
files. It prints `PASS` or `FAIL` and exits with a non-zero status on failure.
It does not assess numerical correctness, report content or word counts, or
execute submitted code.

During preparation, `--draft` permits unanswered `null` fields and skips checking
whether the referenced files exist:

```bash
python check_submission.py results/bb667/results.json --draft
```

Run the full check before submitting. Use tags `<crsid>-q1` through `<crsid>-q4`
to identify your commits for each question without colliding with teammates' tags.

### Repository contents

```text
assignment.pdf              # coursework brief, included in the private handout
README.md                   # setup and submission instructions
requirements.txt            # CPU and submission-check dependencies
check_submission.py         # run before submitting
report/report.tex           # optional report skeleton
results/results.json        # numerical-results template, initially null
results.schema.json         # exact field names and types
src/strata/                 # supplied scientific code
```

The public starter omits `assignment.pdf`; use the brief distributed by the course.
The optional `.gitlab-ci.yml` runs the same check on the private GitLab project.
It requires at least one completed submission and reports incomplete submissions
as failures. Code, tests and plots must also be committed.
