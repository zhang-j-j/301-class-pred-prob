# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune polynomial SVM models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls
load(here("attempt_1/data_splits/keep_wflow_grid.rda"))

# linear recipe
load(here("attempt_1/recipes/lm_rec.rda"))

# Model specification ----
svm_poly_spec <- svm_poly(
  cost = tune(), 
  degree = tune(), 
  scale_factor = tune()
) |> 
  set_engine("kernlab") |> 
  set_mode("classification")

# Define workflow ----
svm_poly_wflow <- workflow() |> 
  add_model(svm_poly_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
svm_poly_params <- extract_parameter_set_dials(svm_poly_spec) |> 
  update(
    cost = cost(),
    degree = degree(),
    scale_factor = scale_factor()
  )

# build tuning grid
svm_poly_grid <- grid_regular(
  svm_poly_params, 
  levels = 3
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
svm_poly_tuned <- svm_poly_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = svm_poly_grid,
    control = keep_wflow_grid
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(svm_poly_tuned, file = here("attempt_1/results/svm_poly_tuned.rda"))
