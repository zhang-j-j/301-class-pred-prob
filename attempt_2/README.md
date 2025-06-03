## Attempt 2

This folder contains all work for attempt 2. The goal of this attempt was to focus on the top-performing model types from attempt 1 and introduce some additional feature engineering steps. These results gave a noticeable improvement upon attempt 1 to meet the top performance benchmark, and the `lightgbm` boosted tree models performed the best.

### Subdirectories

- [`data_splits/`](data_splits): Contains split datasets, folded data, and resampling controls/metrics
- [`recipes/`](recipes): Contains all preprocessing recipes
- [`results/`](results): Contains results from fitting/tuning models with resamples
- [`submissions/`](submissions): Contains final workflows/models and submissions

### R Scripts

- `1_initial_setup.R`: Data splitting, data folding, set up resampling controls
- `2_recipes.R`: Define preprocessing recipes
- `3_tune_btl.R`: Tune boosted tree models (`lightgbm`) with resamples
- `3_tune_btx.R`: Tune boosted tree models (`xgboost`) with resamples
- `3_tune_rf.R`: Tune random forest models (`ranger`) with resamples
- `4_model_comparison.R`: Compare model performance, select final models, analyze hyperparameter tuning values
- `5_fit_submissions.R`: Fit final models to full training set
- `6_final_predictions.R`: Compute predictions on the testing set for submissions
