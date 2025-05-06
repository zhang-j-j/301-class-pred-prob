# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune radial basis function sVM models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls and metrics
load(here("attempt_1/data_splits/keep_wflow_grid.rda"))
load(here("attempt_1/data_splits/my_metrics.rda"))

# linear recipe
load(here("attempt_1/recipes/lm_rec.rda"))

# Model specification ----
svm_rbf_spec <- svm_rbf(
  cost = tune(), 
  rbf_sigma = tune()
) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

# Define workflow ----
svm_rbf_wflow <- workflow() |> 
  add_model(svm_rbf_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
svm_rbf_params <- extract_parameter_set_dials(svm_rbf_spec) |> 
  update(
    cost = cost(),
    rbf_sigma = rbf_sigma()
  )

# build tuning grid
svm_rbf_grid <- grid_regular(
  svm_rbf_params, 
  levels = 5
)

# Fit workflows ----

# set up parallel network sockets
cores <- parallel::detectCores(logical = FALSE) - 1
c1 <- makePSOCKcluster(cores)
registerDoParallel(c1)

# fit workflow
svm_rbf_tuned <- svm_rbf_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = svm_rbf_grid,
    control = keep_wflow_grid,
    metrics = my_metrics
  )

# reset to sequential processing
stopCluster(c1)
registerDoSEQ()
rm(c1)

# Write out results ----
save(svm_rbf_tuned, file = here("attempt_1/results/svm_rbf_tuned.rda"))
