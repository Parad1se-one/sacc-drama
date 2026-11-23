# Reviewer guide

This guide maps the revised manuscript to the included evidence. Start with the summaries or images below; source code and per-episode records are available when a closer inspection is useful.

## Paper-to-evidence index

| Paper item | First file to inspect | What it contains |
| --- | --- | --- |
| Table 6 — Easy | [Easy summary](workspaces/main_eval/experiments/mahrm_direct_20260924/summaries/table6_mahrm_t035_easy_holdout_v1.json) | Seven methods; seed points, means/sample SD, and paired statistics |
| Table 6 — Hard | [Hard summary](workspaces/main_eval/experiments/mahrm_direct_20260924/summaries/table6_mahrm_t035_hard_holdout_v1.json) | The same seven-method protocol on Hard |
| Table 6 — Large | [Large summary](workspaces/large/experiments/summaries/reviewer3_large_random_tempered_sensitivity_v1.json) | Full/all-off at T = 0.35; see the historical-label mapping below |
| Table 6 — Cross-5 and workflow analysis | [Cross summary](workspaces/cross/experiments/summaries/workflow_cross5_randomtrain_v2_final_eval_integrated.json) | Full/team-RM/all-off; select `execution_mode = tempered` for T = 0.35. Table 6 shows Full/all-off; the text additionally discusses team-RM |
| Table 7 — component analysis | [Per-seed ablation values](figures/tenseed_ablation_seed_summary.csv) | AUC and selected-checkpoint Completion, Success, and Steps on the selection set |
| Figure 3 — environments and workflows | [PNG](figures/environment_workflow_structure.png), [plotting program](figures/plot_environment_workflows.py) | Easy, Hard, Large, and Cross layouts and precedence structures |
| Figure 4 — full Completion training curves | [PNG](figures/completion_easy_hard_5seed.png), [observations](figures/figure_sources/completion_training_observations.csv), [plotting program](figures/plot_completion_curves.py) | All 60 six-method training runs, 126,000 observations; Easy 3,000 / Hard 1,200 episodes |
| Figure 5 — ten-seed component analysis | [PNG](figures/ablation_seed_results.png), [selection curves](figures/tenseed_ablation_selection_curves.json), [plotting program](figures/plot_ablation_seed_results.py) | All five configurations and ten training seeds per difficulty |
| MAHRM direct comparison | [English method note](MAHRM_METHOD.md) | Hierarchical RM implementation, fixed-role adaptation, and evaluation protocol |

Each Table 6 JSON has a CSV with the same base filename alongside it. The [evaluation index](evaluation_index.json) connects each of the 94 evaluated policies/modes to its run record, raw episodes, test starts, and checkpoint hash. It uses normalized method labels throughout. The [ablation audit](figures/figure_sources/ablation_seed_results_audit_20260927.json) records the Table 7 consistency checks and statistical analysis.

## Protocols at a glance

Seed counts below refer to independently trained models **per method/configuration**, not to the number of test episodes.

| Evidence | Training seeds | Training budget (episodes) | Evaluation data and step cap | Action rule / reported quantity |
| --- | --- | --- | --- | --- |
| Table 6 Easy | 5: 42–46 | 3,000 | 100 shared held-out random starts; 300 steps | Actors T = 0.35; QMIX/VDN/MAHRM greedy; held-out performance |
| Table 6 Hard | 5: 42–46 | 1,200 | 100 shared held-out random starts; 500 steps | Same action rules; held-out performance |
| Table 6 Large | 3: 42, 123, 456 | 4,000; fixed training starts | 100 shared held-out random starts; 800 steps | Full/all-off actors T = 0.35 |
| Table 6 / text Cross-5 | 3: 42, 123, 456 | 4,000; random training starts | 100 shared held-out random starts; 800 steps | Full/team-RM/all-off actors T = 0.35; greedy results kept as a separate mode |
| Table 7 / Figure 5 ablation | 10; listed in the CSV | Easy 1,000 / Hard 3,000; fixed training starts | 20 selection starts; 300/500 steps | Selection-curve normalized AUC and selected-checkpoint selection scores |
| Figure 4 training curves | 5: 42–46 | Easy 3,000 / Hard 1,200 | Online training episodes under each method's exploration policy | Training Completion, not held-out test performance |

The ablation's Hard budget differs from Table 6's Hard budget because these are separate experimental protocols. AUC uses trapezoidal integration over 0–1,000 / 0–3,000 episodes, normalized by the horizon, with the fixed boundary point (0, 0). For Table 7, read the `selected_*` CSV columns, not the `final_*` columns. MAHRM records periodic selection scores rather than per-episode training Completion, so it appears in Table 6 but not Figure 4.

Checkpoint selection uses the recorded selection sets and rules, separate from the held-out tests. The seven-method comparison retains the SACC and neural baselines' recorded checkpoint-selection protocols; MAHRM selects every 200 episodes on 20 selection starts. Exact intervals and selected episodes are preserved in the specifications and run records. The Cross greedy and T = 0.35 rows are not extra independent training seeds and should not be pooled.

## Method labels

| Reader-facing name | Where / stored label | Meaning |
| --- | --- | --- |
| SACC (Full) | Easy/Hard `sacc`; Large `sacc`; Cross `full` | Full progress inputs and structured rewards |
| all-off | Large historical `happo`; Cross/ablation `all_off` | The SACC optimization backbone with actor/critic progress inputs and structured rewards removed |
| HAPPO | Easy/Hard `happo` | The separately trained HAPPO baseline; **not** the Large all-off label |
| team-RM | Cross `team_rm` | Shared team-progress representation; distinct from MAHRM |
| MAHRM adaptation | Easy/Hard `mahrm` | Independently implemented hierarchical RM baseline adapted to DRAMA's fixed roles |

The Large summary retains its original `sacc`/`happo` field values for traceability. Use Full/all-off when interpreting those two rows. The normalized [evaluation index](evaluation_index.json) already labels them `full`/`all_off`; no original run record has been relabeled or overwritten.

## Metrics and scope

Completion is the fraction of completed stage instances; Success indicates completion of the entire task within the cap. Per-seed metrics average the 100 test episodes, and reported means/sample SD then aggregate the independent training seeds. For comparisons across methods, use `mean_environment_return`; `mean_objective_return` may include method-specific reward shaping.

This repository contains the evidence indexed above for the protocols in the manuscript. Older protocol files retained within source snapshots document their dependencies; they do not define additional Table 6 rows. Use the linked summaries and evaluation index to identify the reported comparisons. Different execution modes and selection-set metrics remain separately labeled.

## Three levels of reproducibility

1. **Inspect:** open the PNGs, CSVs, and JSON summaries without installing software.
2. **Verify saved evidence:** run `python verify_results.py` from a clean extraction with Python 3.10+. This requires no GPU, third-party packages, or weights.
3. **Execute research code:** use the appropriate source workspace and a separate working copy, following [README.md](README.md). Weights are not bundled; policy evaluation requires the corresponding checkpoints. Some historical orchestration scripts expect the original Git checkout.

File manifests and original/exported hashes support traceability; they are not required reading to understand the paper's results.
