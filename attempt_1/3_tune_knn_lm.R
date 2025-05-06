# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune knn models with linear recipe

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(2246)

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls and metrics
load(here("attempt_1/data_splits/keep_wflow_grid.rda"))
load(here("attempt_1/data_splits/my_metrics.rda"))

# linear recipe
load(here("attempt_1/recipes/lm_rec.rda"))

# Model specification ----
knn_lm_spec <- nearest_neighbor(neighbors = tune()) |> 
  set_engine("kknn") |> 
  set_mode("regression")

# Define workflow ----
knn_lm_wflow <- workflow() |> 
  add_model(knn_lm_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
knn_lm_params <- extract_parameter_set_dials(knn_lm_spec) |> 
  update(
    neighbors = neighbors(c(1, 30))
  )

# build tuning grid
knn_lm_grid <- grid_regular(
  knn_lm_params, 
  levels = 15
)

# Fit workflows ----

# set up parallel network sockets
cores <- parallel::detectCores(logical = FALSE) - 1
c1 <- makePSOCKcluster(cores)
registerDoParallel(c1)

# fit workflow
knn_lm_tuned <- knn_lm_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = knn_lm_grid,
    control = keep_wflow_grid,
    metrics = my_metrics
  )

# reset to sequential processing
stopCluster(c1)
registerDoSEQ()
rm(c1)

# Write out results ----
save(knn_lm_tuned, file = here("attempt_1/results/knn_lm_tuned.rda"))
