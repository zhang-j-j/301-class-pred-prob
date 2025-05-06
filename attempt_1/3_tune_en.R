# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: tune elastic net models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls and metrics
load(here("attempt_1/data_splits/keep_wflow_grid.rda"))
load(here("attempt_1/data_splits/my_metrics.rda"))

# linear recipe
load(here("attempt_1/recipes/lm_rec.rda"))

# Model specification ----
en_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) |> 
  set_engine("glmnet") |> 
  set_mode("regression")

# Define workflow ----
en_wflow <- workflow() |> 
  add_model(en_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
en_params <- extract_parameter_set_dials(en_spec) |> 
  update(
    penalty = penalty(c(-5, 0)),
    mixture = mixture()
  )

# build tuning grid
en_grid <- grid_regular(
  en_params, 
  levels = c(penalty = 10, mixture = 6)
)

# Fit workflows ----

# set up parallel network sockets
cores <- parallel::detectCores(logical = FALSE) - 1
c1 <- makePSOCKcluster(cores)
registerDoParallel(c1)

# fit workflow
en_tuned <- en_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = en_grid,
    control = keep_wflow_grid,
    metrics = my_metrics
  )

# reset to sequential processing
stopCluster(c1)
registerDoSEQ()
rm(c1)

# Write out results ----
save(en_tuned, file = here("attempt_1/results/en_tuned.rda"))
