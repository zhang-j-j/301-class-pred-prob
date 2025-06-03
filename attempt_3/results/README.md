## Results

This subdirectory contains fitted sub-models for tuning hyperparameters. The large objects can be accessed via [Google Drive](https://drive.google.com/drive/folders/1u8n32uvsgk1MbWuc_N_dgInZDa2Sj92B?usp=sharing).

- [`ensemble/`](ensemble): Contains results from ensemble model stacks (`4_train_ensemble.R`)
- `bt_tuned.rda`: Results from tuning a boosted tree model with resamples (`3_tune_bt.R`)
- `en_tuned.rda`: Results from tuning an elastic net model with resamples (`3_tune_en.R`)
- `knn_tuned.rda`: Results from tuning a K-nearest neighbors model with resamples (`3_tune_knn.R`)
- `log_fit.rda`: Results from fitting a logistic regression model to resamples (`3_fit_log.R`)
- `mars_tuned.rda`: Results from tuning a MARS model with resamples (`3_tune_mars.R`)
- `nn_tuned.rda`: Results from tuning a neural network model with resamples (`3_tune_nn.R`)
- `svm_tuned.rda`: Results from tuning an SVM model (radial basis function kernel) with resamples (`3_tune_svm.R`)
