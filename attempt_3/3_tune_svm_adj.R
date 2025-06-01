# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune radial basis function svm models (adjusted preprocessing trial)

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

# svm recipe
load(here("attempt_3/recipes/svm_rec.rda"))

# Model specification ----
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) |> 
  set_engine("kernlab") |> 
  set_mode("classification")

# Define workflow ----
svm_wflow <- workflow() |> 
  add_model(svm_spec) |> 
  add_recipe(svm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
svm_params <- extract_parameter_set_dials(svm_wflow) |> 
  update(
    cost = cost(c(-3, 7)),
    rbf_sigma = rbf_sigma(c(-5, 0))
  )

# # build tuning grid
# svm_grid <- grid_regular(
#   svm_params,
#   levels = c(cost = 6, rbf_sigma = 6)
# )
svm_grid <- tibble(
  cost = c(2, 2, 8, 8, 32, 32),
  rbf_sigma = c(0.001, 0.01, 0.001, 0.01, 0.001, 0.01)
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
svm_tuned_adj <- svm_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = svm_grid,
    control = control_stack_grid()
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(svm_tuned_adj, file = here("attempt_3/results/svm_tuned_adj.rda"))
