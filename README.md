# SACC and DRAMA

Code and reproducibility materials for *Learning Stage-Aware Construction Coordination for Heterogeneous Construction Robots*. DRAMA is the research simulator developed alongside SACC. [Paper on OpenReview](https://openreview.net/forum?id=Wjf2Rs44Zv)

## Reproduce the paper

The [reproducibility package](reproducibility/README.md) contains the experiment-specific source code, frozen evaluation protocols, results, and figure scripts. The [table and figure guide](reproducibility/REVIEWER_GUIDE.md) maps each result to its files.

To verify the recorded results from a fresh clone, run:

```sh
git clone https://github.com/Parad1se-one/sacc-drama.git
cd sacc-drama/reproducibility
python3 verify_results.py
```

The verifier uses the Python 3.10+ standard library. For training, use the source snapshot and dependencies identified in the [package instructions](reproducibility/README.md).

The top-level `DRAMA/` and `SACC/` directories preserve the original code release. The experiments reported in the paper use the snapshots under [reproducibility/workspaces](reproducibility/workspaces).
