## Attempt 3

This folder contains all work for attempt 3. The goal of this attempt was to try an ensemble model using the best tree-based model from attempt 2 and simpler model types from attempt 1 with some additional feature engineering. These results did not appear to improve upon attempt 2, though the top performance benchmark had already been met.

### Subdirectories

- [`data_splits/`](data_splits): Contains split datasets, folded data, and resampling controls/metrics
- [`recipes/`](recipes): Contains all preprocessing recipes
- [`results/`](results): Contains results from fitting/tuning models with resamples
- [`submissions/`](submissions): Contains final workflows/models and submissions

### R Scripts

- `1_initial_setup.R`: Data splitting, data folding, set up resampling controls
- `2_recipes.R`: Define preprocessing recipes
- `3_fit_log.R`: Fit logistic regression models to resamples
- `3_tune_bt.R`: Tune boosted tree models with resamples
- `3_tune_en.R`: Tune elastic net models with resamples
- `3_tune_knn.R`: Tune K-nearest neighbors models with resamples
- `3_tune_mars.R`: Tune multivariate adaptive regression splines (MARS) models with resamples
- `3_tune_nn.R`: Tune multilayer perceptron (neural network) models with resamples
- `3_tune_rf.R`: Tune random forest models with resamples
- `3_tune_svm.R`: Tune support vector machine (SVM) models with resamples (radial basis function kernel)
- `4_model_comparison.R`: Compare model performance, select and fit final models, analyze hyperparameter tuning values
- `4_train_ensemble.R`: Define and fit ensemble model stacks
- `5_assess_models.R`: Assess fitted models using predictions on the testing set
- `6_fit_submissions`: Fit final models to full training set
- `7_final_predictions.R`: Compute predictions on the testing set for submissions
