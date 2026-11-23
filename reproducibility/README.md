# Learning Stage-Aware Construction Coordination (SACC)

Code and reproducibility materials for *Learning Stage-Aware Construction Coordination for Heterogeneous Construction Robots* ([TMLR / OpenReview, submission #9837](https://openreview.net/forum?id=Wjf2Rs44Zv)). DRAMA is a research simulator developed alongside SACC.

## Start here

| Your goal | Start with | Requirements |
| --- | --- | --- |
| Read the evidence behind a paper table or figure | [Reviewer guide: figure/table index and protocol overview](REVIEWER_GUIDE.md) | No installation |
| Check the included results and file integrity | `python verify_results.py` from this directory | Python 3.10+ standard library only; no GPU, package installation, or model weights |
| Understand the RM baseline | [MAHRM adaptation and evaluation protocol](MAHRM_METHOD.md) | No installation |
| Run source tests or training code | The optional instructions below, in a separate working copy | Research dependencies; training may require a GPU |

The primary held-out comparisons use actor temperature T = 0.35; QMIX, VDN, and the tabular MAHRM adaptation act greedily. Cross also includes a separately labeled greedy evaluation. Online training curves and the ten-seed selection-set ablation are distinct from these held-out tests.

## 1. Verify the included evidence

From a fresh clone:

```bash
git clone https://github.com/Parad1se-one/sacc-drama.git
cd sacc-drama/reproducibility
python3 verify_results.py
```

No virtual environment or `pip install` is needed for this check. A successful run prints JSON with `"status": "passed"`, 94 evaluations, 9,400 raw episodes, 100 ablation seed curves, and 126,000 full-training observations. The verifier checks the file manifest, ordered test starts, per-episode bounds, seed-level means/sample SD, ablation AUC, and full-length training data. It does not train models or launch new evaluations.

Run the check from the reproducibility directory of a clean clone or GitHub source archive. Keep virtual environments and newly generated outputs outside this copy: integrity verification intentionally reports added, missing, or changed files. Root Git metadata and Python and pytest caches are ignored. If a check fails, use a fresh clone or source archive rather than changing the supplied manifest.

## 2. Find code and data

| Workspace | Purpose |
| --- | --- |
| [main_eval](workspaces/main_eval) | Table 6 Easy/Hard: seven methods, five seeds, 70 evaluations / 7,000 raw episodes; includes MAHRM code |
| [sacc_train](workspaces/sacc_train) | SACC training implementation and five-seed training specifications |
| [baseline_train](workspaces/baseline_train) | QMIX, VDN, MAPPO, IPPO, and HAPPO training implementation and records |
| [large](workspaces/large) | Table 6 Large: Full/all-off, three seeds, six evaluations / 600 raw episodes |
| [cross](workspaces/cross) | Table 6 and workflow analysis: Full/team-RM/all-off, three seeds, greedy and T = 0.35; 18 evaluations / 1,800 raw episodes |
| [ablation](workspaces/ablation) | Table 7 / Figure 5: five configurations, ten seeds per difficulty, fixed-training-start component analysis |
| [figures](figures) | Figures 3–5, plotting programs, and numerical inputs |

These are separate source snapshots of the implementations used in each experiment, not six packages to install together. Choose the workspace for the experiment of interest. The [reviewer guide](REVIEWER_GUIDE.md) links directly to the relevant summaries and explains method labels.

This package excludes model weights, machine logs, reviewer comments, and author-only drafts. **The recorded numerical results can be checked immediately; rerunning policy evaluation requires retraining or otherwise obtaining the corresponding checkpoints.** Checkpoint hashes identify the original selected weights but do not substitute for them.

## 3. Optional: source tests, training, and figures

First verify the clean extraction. For execution, make a separate copy and create its virtual environment beside that copy, not inside it. For example, from the directory containing the clone on a Unix-like system (use unused destination names):

```bash
cp -R sacc-drama/reproducibility SACC_working_copy
python -m venv SACC_runtime
. SACC_runtime/bin/activate
python -m pip install -r SACC_working_copy/requirements.txt
cd SACC_working_copy/workspaces/cross
PYTHONPATH=src/sacc-drama/DRAMA:src/sacc-drama/SACC python -m pytest -q src/sacc-drama/DRAMA/tests src/sacc-drama/SACC/tests
```

The same test command applies in the other workspaces. [RUNTIME.json](RUNTIME.json) records the experiment environment; [requirements.txt](requirements.txt) lists direct dependencies, not a fully pinned environment lock. These dependencies are for source execution, not the standard-library result verifier.

Training records contain the resolved arguments and seeds. From the working-copy root, this command prints, but does not execute, an example training command:

```bash
python record_command.py workspaces/sacc_train/experiments/run_records/submission_reproduction_easy_v1/submission_reproduction_easy_v1__submitted_sacc__seed42.json
```

Review the printed command and output paths before running it. Original orchestration scripts are retained for provenance; some require the original Git checkout and are not a one-command entry point for this repository. MAHRM has its own `train` and `eval` entry points, described in [MAHRM_METHOD.md](MAHRM_METHOD.md).

For Figures 3–5, run the corresponding program from the working copy's `figures` directory: `python plot_environment_workflows.py`, `python plot_completion_curves.py`, or `python plot_ablation_seed_results.py`. The included PNGs can be viewed without executing code. Full Easy/Hard curves retain 3,000/1,200 episodes; the plot applies the same EMA smoothing to every seed and shows sample SD, not a confidence interval.

## 4. Interpretation and provenance

The statistical unit is an independent training seed. Each seed's 100 test episodes estimate its performance; they are not 100 independent training replicates. Cross-method reward comparisons use `mean_environment_return`, not shaped objective return. Seed points, sample SD, paired effects, exact sign-flip tests, and the applicable bootstrap/IQM summaries are retained in the linked evidence. The ablation uses selection-set learning efficiency and includes Holm correction for its four component-removal comparisons.

SACC has the highest mean Completion and Success in the seven-method Easy/Hard comparison, and Full leads Cross at T = 0.35. The small seed counts do not establish significant superiority over every baseline. The complete training curves retain the Hard late-training decline. Configuration rankings depend on the execution protocol; the results support the manuscript's protocol-specific claims.

For individual runs, use [evaluation_index.json](evaluation_index.json). [MANIFEST_SHA256.json](MANIFEST_SHA256.json) covers the exported files; [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records original and exported hashes. `{ARTIFACT_ROOT}` denotes this extracted directory; `{ARCHIVED_ROOT}` denotes an unbundled historical location. Hashes inside historical records refer to their original source bytes, whereas the manifest covers the path-relocated review copy. Numerical evidence is unchanged.

## Attribution

The SACC/DRAMA code retains its original [MIT license and copyright notice](LICENSE). The six source snapshots preserve their respective license files and notices. This public repository is derived from the submission supplement and retains its frozen numerical evidence.
