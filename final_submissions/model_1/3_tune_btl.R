# Classification Prediction Problem ----
# Stat 301-3
# Final model 1
# Step 3: tune boosted tree models (lightgbm)

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(392)

# Load objects ----

# resamples
load(here("final_submissions/model_1/data_splits/airbnb_folds.rda"))

# controls
load(here("final_submissions/model_1/data_splits/keep_wflow_grid.rda"))

# tree recipe
load(here("final_submissions/model_1/recipes/tree_rec.rda"))

# Model specification ----
bt_spec <- boost_tree(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  learn_rate = tune()
) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

# Define workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(tree_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
# from the recipe, 95 predictor columns after preprocessing
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(
    mtry = mtry(c(1, 90)),
    trees = trees(c(100, 1000)),
    min_n = min_n(),
    learn_rate = learn_rate(c(-4, -0.1))
  )

# build tuning grid
bt_grid <- grid_regular(
  bt_params,
  levels = c(mtry = 3, trees = 4, min_n = 3, learn_rate = 8)
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
btl_tuned <- bt_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = bt_grid,
    control = keep_wflow_grid
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(btl_tuned, file = here("final_submissions/model_1/results/btl_tuned.rda"))
