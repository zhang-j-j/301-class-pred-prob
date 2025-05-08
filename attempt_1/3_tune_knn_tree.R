# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune knn models with tree recipe

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(419)

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls
load(here("attempt_1/data_splits/keep_wflow_grid.rda"))

# linear recipe
load(here("attempt_1/recipes/tree_rec.rda"))

# Model specification ----
knn_tree_spec <- nearest_neighbor(neighbors = tune()) |> 
  set_engine("kknn") |> 
  set_mode("classification")

# Define workflow ----
knn_tree_wflow <- workflow() |> 
  add_model(knn_tree_spec) |> 
  add_recipe(tree_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
knn_tree_params <- extract_parameter_set_dials(knn_tree_spec) |> 
  update(
    neighbors = neighbors(c(1, 30))
  )

# build tuning grid
knn_tree_grid <- grid_regular(
  knn_tree_params, 
  levels = 15
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
knn_tree_tuned <- knn_tree_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = knn_tree_grid,
    control = keep_wflow_grid
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(knn_tree_tuned, file = here("attempt_1/results/knn_tree_tuned.rda"))
