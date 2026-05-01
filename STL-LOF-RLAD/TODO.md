# TODO

- Original Ensemble-RLAD has an archived script, but no independently audited JSON result matching the locked final protocol was found. It is marked `not available` in `results_final/baseline_results.csv`.
- Average precision was not reported in the available baseline comparison JSON. AP fields are marked `not available` rather than inferred.
- Final-protocol ablation results for `Without Active Learning`, `Without LOF using 3sigma`, and `Without STL using LOF on raw data` were not found. Older ablation reports exist but use different data or stride settings, so they were not promoted into the final table.
- `fig8_training_dynamics.tif` needs episode-level training history from the locked final run. The archived locked run did not include that history, so the generated figure is an unavailable-data audit panel.
- `fig14_sensitivity.tif` needs `results_final/sensitivity_results.csv`. No final-protocol sensitivity file was found, so the generated figure is an unavailable-data audit panel.
- Raw underground coal mine monitoring data are confidential and should not be uploaded publicly.
- If a collaborator reruns training locally, data paths in `config/final_config.json` or legacy archived scripts may need local adjustment.
- Before any future rerun is used in the manuscript, verify chronological splitting happens before normalization and LOF threshold estimation.
