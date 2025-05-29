# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune radial basis function svm models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
library(future)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# resamples
load(here("attempt_3/data_splits/airbnb_folds.rda"))

# controls and metrics
load(here("attempt_3/data_splits/my_metrics.rda"))

# tree recipe
load(here("attempt_3/recipes/lm_rec.rda"))

# Model specification ----
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

# Define workflow ----
svm_wflow <- workflow() |> 
  add_model(svm_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
svm_params <- extract_parameter_set_dials(svm_wflow) |> 
  update(
    cost = cost(c(-3, 7)),
    rbf_sigma = rbf_sigma(c(-5, 0))
  )

# build tuning grid
svm_grid <- grid_regular(
  svm_params,
  levels = c(cost = 6, rbf_sigma = 6)
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
svm_tuned <- svm_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = svm_grid,
    control = control_stack_grid(),
    metrics = my_metrics
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(svm_tuned, file = here("attempt_3/results/svm_tuned.rda"))
