## Attempt 1

This folder contains all work for attempt 1. The goal of this attempt was to complete an initial screening stage for many different model types with only the minimal required preprocessing for each. All of the proposed model types were run successfully in this attempt, and these initial results indicated that tree-based models performed the best.

### Subdirectories

- [`data_splits/`](data_splits): Contains split datasets, folded data, and resampling controls/metrics
- [`recipes/`](recipes): Contains all preprocessing recipes
- [`results/`](results): Contains results from fitting/tuning models with resamples
- [`submissions/`](submissions): Contains final workflows/models and submissions

### R Scripts

- `1_initial_setup.R`: Data splitting, data folding, set up resampling controls
- `2_recipes.R`: Define preprocessing recipes
- `3_fit_log.R`: Fit logistic regression models to resamples
- `3_fit_nb.R`: Fit naive bayes models to resamples
- `3_tune_bt.R`: Tune boosted tree models with resamples
- `3_tune_en.R`: Tune elastic net models with resamples
- `3_tune_knn_lm.R`: Tune K-nearest neighbors models with resamples (linear model recipe)
- `3_tune_knn_tree.R`: Tune K-nearest neighbors models with resamples (tree-based recipe)
- `3_tune_mars.R`: Tune multivariate adaptive regression splines (MARS) models with resamples
- `3_tune_nn.R`: Tune multilayer perceptron (neural network) models with resamples
- `3_tune_rf.R`: Tune random forest models with resamples
- `3_tune_svm_rbf.R`: Tune support vector machine (SVM) models with resamples (radial basis function kernel)
- `4_model_comparison.R`: Compare model performance, select final models, analyze hyperparameter tuning values
- `5_fit_submissions.R`: Fit final models to full training set
- `6_final_predictions.R`: Compute predictions on the testing set for submissions
