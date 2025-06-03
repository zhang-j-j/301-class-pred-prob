## Model 2

Model 2 was selected from attempt 3. To reproduce these results, simply run the scripts sequentially in the indicated order (1-5).

### Subdirectories

- [`data_splits/`](data_splits): Contains split datasets, folded data, and resampling controls/metrics
- [`recipes/`](recipes): Contains all preprocessing recipes
- [`results/`](results): Contains all results from model 2

### R Scripts

- `1_initial_setup.R`: Data splitting, data folding, set up resampling controls
- `2_recipes.R`: Define preprocessing recipes
- `3_tune_bt.R`: Tune boosted tree models with resamples
- `4_model_comparison.R`: Compare model performance, select and fit final models, analyze hyperparameter tuning values
- `5_final_predictions.R`: Compute predictions on the testing set for submissions
