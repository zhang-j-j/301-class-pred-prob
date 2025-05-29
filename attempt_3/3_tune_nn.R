# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune neural network models

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

# tree recipe
load(here("attempt_3/recipes/tree_rec.rda"))

# Model specification ----
nn_spec <- mlp(
  hidden_units = tune(),
  penalty = tune()
) |> 
  set_engine("nnet", MaxNWts = 5000) |> 
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
    control = control_stack_grid()
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(nn_tuned, file = here("attempt_3/results/nn_tuned.rda"))
