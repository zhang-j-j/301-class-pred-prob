# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune boosted tree models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(10)

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls and metrics
load(here("attempt_1/data_splits/keep_wflow_grid.rda"))
load(here("attempt_1/data_splits/my_metrics.rda"))

# linear recipe
load(here("attempt_1/recipes/tree_rec.rda"))

# Model specification ----
bt_spec <- boost_tree(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  learn_rate = tune()
) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# Define workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(tree_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
# from the recipe, 39 predictor columns after preprocessing
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(
    mtry = mtry(c(1, 35)),
    trees = trees(c(10, 700)),
    min_n = min_n(),
    learn_rate = learn_rate(c(-5, -0.2))
  )

# build tuning grid
bt_grid <- grid_regular(
  bt_params, 
  levels = c(mtry = 3, trees = 4, min_n = 3, learn_rate = 8)
)

# Fit workflows ----

# set up parallel network sockets
cores <- parallel::detectCores(logical = FALSE) - 1
c1 <- makePSOCKcluster(cores)
registerDoParallel(c1)

# fit workflow
bt_tuned <- bt_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = bt_grid,
    control = keep_wflow_grid,
    metrics = my_metrics
  )

# reset to sequential processing
stopCluster(c1)
registerDoSEQ()
rm(c1)

# Write out results ----
save(bt_tuned, file = here("attempt_1/results/bt_tuned.rda"))
