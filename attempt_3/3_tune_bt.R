# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune boosted tree models (lightgbm)

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
library(future)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(317)

# Load objects ----

# resamples
load(here("attempt_3/data_splits/airbnb_folds.rda"))

# tree recipe
load(here("attempt_3/recipes/tree_rec.rda"))

# Model specification ----
bt_spec <- boost_tree(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  learn_rate = tune(),
  tree_depth = tune()
) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

# Define workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(tree_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
# from the recipe, 98 predictor columns after preprocessing
# adjust based on attempt 2 results ----
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(
    mtry = mtry(c(40, 90)),
    trees = trees(c(600, 1200)),
    min_n = min_n(c(2, 20)),
    learn_rate = learn_rate(c(0.03, 0.09), trans = NULL),
    tree_depth = tree_depth()
  )

# build tuning grid
bt_grid <- grid_regular(
  bt_params,
  levels = c(mtry = 3, trees = 3, min_n = 3, learn_rate = 3, tree_depth = 4)
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
bt_tuned <- bt_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = bt_grid,
    control = control_stack_grid()
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(bt_tuned, file = here("attempt_3/results/bt_tuned.rda"))
