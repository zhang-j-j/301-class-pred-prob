# Classification Prediction Problem ----
# Stat 301-3
# Attempt 2
# Step 3: tune random forest models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(16)

# Load objects ----

# resamples
load(here("attempt_2/data_splits/airbnb_folds.rda"))

# controls
load(here("attempt_2/data_splits/keep_wflow_grid.rda"))

# tree recipe
load(here("attempt_2/recipes/tree_rec.rda"))

# Model specification ----
rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500, # don't tune trees this time
  min_n = tune()
) |> 
  set_engine("ranger") |> 
  set_mode("classification")

# Define workflow ----
rf_wflow <- workflow() |> 
  add_model(rf_spec) |> 
  add_recipe(tree_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
# from the recipe, 95 predictor columns after preprocessing
rf_params <- extract_parameter_set_dials(rf_spec) |> 
  update(
    mtry = mtry(c(1, 90)),
    min_n = min_n()
  )

# build tuning grid
rf_grid <- grid_regular(
  rf_params, 
  levels = c(mtry = 5, min_n = 5)
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
rf_tuned <- rf_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = rf_grid,
    control = keep_wflow_grid
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(rf_tuned, file = here("attempt_2/results/rf_tuned.rda"))
