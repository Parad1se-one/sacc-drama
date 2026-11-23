# MAHRM adaptation for DRAMA

This comparison uses an independent implementation based on Zheng and Yu, *Multi-Agent Reinforcement Learning with a Hierarchy of Reward Machines* (2024), https://arxiv.org/abs/2403.07005. It adapts the hierarchical RM construction to DRAMA's four fixed construction roles. It is not an author-provided implementation. The Cross team-RM control is a separate configuration, not MAHRM.

## Hierarchy and action selection

The top-level proposition is completion of all construction cells by the team. Each cell progresses through four stages, represented by states 0 → 1 → 2 → 3 → 4. Bottom-level events are successful `distribute_floor`, `vibrate_floor`, `level_floor`, and `cover_surface` actions reported by the DRAMA environment's `step` function.

At an option boundary, the upper level assigns each of the four robots a construction cell that can advance its current RM task. An option terminates when its selected lower-level RM transition occurs or after 50 steps. Upper-level updates account for the option duration through discounting. Lower-level policies use tabular QRM with legal-action masks and counterfactual updates for cell RMs belonging to the same role. The frozen parameters are ε = 0.1, α = 0.1, γ = 0.9, and a maximum option duration of 50 steps.

DRAMA's four construction roles are not interchangeable. This adaptation therefore restricts eligible subtasks by role and represents upper-level joint option values using role-factorized Q values, rather than allowing arbitrary agent grouping. This is the principal structural difference between the DRAMA comparison and the original method's setting. Training uses event-based RM rewards; cross-method evaluation uses the unshaped environment return.

## Training, selection, and held-out evaluation

| Setting | Training seeds | Episodes | Episode cap | Checkpoint selection | Final evaluation |
| --- | --- | --- | --- | --- | --- |
| Easy | 42–46 | 3,000 | 300 steps | Every 200 episodes on 20 selection starts | 100 held-out random starts |
| Hard | 42–46 | 1,200 | 500 steps | Every 200 episodes on 20 selection starts | 100 held-out random starts |

Checkpoint selection maximizes mean Completion on the selection set, with Success used to break ties. The test set is separate. For each difficulty, all seven methods in Table 6 use the same ordered 100 test starts. The frozen [protocol audit](workspaces/main_eval/experiments/mahrm_direct_20260924/protocols/protocol_audit.json) records the start-set separation checks.

SACC, MAPPO, IPPO, and HAPPO actors use T = 0.35. QMIX, VDN, and this tabular MAHRM adaptation use greedy actions; a softmax actor temperature does not apply to these greedy evaluations. SACC and the neural baselines retain their recorded checkpoint-selection protocols, while the MAHRM rule is specified above.

## Results and interpretation

MAHRM's mean Completion is 0.4080 on Easy and 0.4169 on Hard, with zero Success on both. The full seven-method results, sample SD, seed points, IQM, and paired statistical comparisons are in the [Easy summary](workspaces/main_eval/experiments/mahrm_direct_20260924/summaries/table6_mahrm_t035_easy_holdout_v1.json) and [Hard summary](workspaces/main_eval/experiments/mahrm_direct_20260924/summaries/table6_mahrm_t035_hard_holdout_v1.json).

The SACC-minus-MAHRM Completion differences are 0.57725 and 0.58020 on Easy and Hard; the two-sided exact sign-flip p-value is 0.0625 for each five-seed comparison. These results characterize the fixed-role DRAMA adaptation. They do not establish significant superiority over the original MAHRM method across environments. Independent training seeds, rather than the 100 episodes within each seed, are the units for uncertainty estimates.

## Implementation and evidence

- [mahrm_drama.py](workspaces/main_eval/experiments/mahrm_direct_20260924/code/mahrm_drama.py): implementation and `train` / `eval` commands.
- [Frozen protocols](workspaces/main_eval/experiments/mahrm_direct_20260924/protocols): training, selection, and test starts.
- [Evaluation specifications](workspaces/main_eval/experiments/mahrm_direct_20260924/specs): the shared seven-method comparison.
- [evaluation_index.json](evaluation_index.json): each run's raw episodes, start set, and checkpoint hash.

The shared comparison comprises 70 evaluations and 7,000 raw episodes. Run `python verify_results.py` from the archive root to verify the saved evidence without installing research dependencies or loading model weights. To execute the implementation, use a separate working copy and the environment instructions in [README.md](README.md). The `eval` command requires a checkpoint; weights are not distributed in this ZIP.
