# Design package — Scenario-Level Explanations for Health-Care Fraud Detection

| File | What it covers |
|---|---|
| [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md) | The full design: problem, data, architecture, attribution methods, evaluation, audiences and lifecycle, outcomes, limitations, risks, work plan, decision log |
| [FEATURE_CATALOG.md](FEATURE_CATALOG.md) | Every proposed feature by scheme group: formula, source fields, direction, rationale; context group; leakage checklist |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Experiment matrix, splits, metric formulas, injection parameter grid, sensitivity suite, result-table shells, compute estimate |

## Figures

| # | File | Shows |
|---|---|---|
| 1 | [fig01_system_architecture.png](figures/fig01_system_architecture.png) | End-to-end pipeline in seven stages |
| 2 | [fig02_data_model.png](figures/fig02_data_model.png) | Source tables → NPI-year analysis table |
| 3 | [fig03_label_timeline.png](figures/fig03_label_timeline.png) | Data availability, label rule, label horizon, temporal split |
| 4 | [fig04_scenario_taxonomy.png](figures/fig04_scenario_taxonomy.png) | Seven scheme groups + context group |
| 5 | [fig05_attribution_methods.png](figures/fig05_attribution_methods.png) | Five attribution methods (A–E) |
| 6 | [fig06_peer_baseline.png](figures/fig06_peer_baseline.png) | Global vs peer baseline (illustrative) |
| 7 | [fig07_scheme_injection.png](figures/fig07_scheme_injection.png) | Ground-truth scheme injection |
| 8 | [fig08_evaluation_framework.png](figures/fig08_evaluation_framework.png) | Evaluation tests and decision logic |
| 9 | [fig09_audiences_lifecycle.png](figures/fig09_audiences_lifecycle.png) | Audiences and explanation provenance |

Numbers inside Figs. 5 and 6 are illustrative, not results.

To regenerate the figures (requires `matplotlib` and `numpy`), run from this folder:

```bash
python figures/_src/fig_arch.py figures
```

```bash
python figures/_src/fig_method.py figures
```
