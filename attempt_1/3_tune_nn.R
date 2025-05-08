# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune neural network models

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
load(here("attempt_1/recipes/tree_rec.rda"))

# Model specification ----
nn_spec <- mlp(
  hidden_units = tune(),
  penalty = tune()
) |> 
  set_engine("nnet") |> 
  set_mode("classification")

# Define workflow ----
nn_wflow <- workflow() |> 
  add_model(nn_spec) |> 
  add_recipe(tree_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
nn_params <- extract_parameter_set_dials(nn_spec) |> 
  update(
    hidden_units = hidden_units(),
    penalty = penalty()
  )

# build tuning grid
nn_grid <- grid_regular(
  nn_params, 
  levels = 5
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
nn_tuned <- nn_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = nn_grid,
    control = keep_wflow_grid
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(nn_tuned, file = here("attempt_1/results/nn_tuned.rda"))
