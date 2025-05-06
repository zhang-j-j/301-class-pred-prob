# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: fit ordinary least squares models

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
load(here("attempt_1/data_splits/keep_wflow_res.rda"))
load(here("attempt_1/data_splits/my_metrics.rda"))

# linear recipe
load(here("attempt_1/recipes/lm_rec.rda"))

# Model specification ----
ols_spec <- linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression")

# Define workflow ----
ols_wflow <- workflow() |> 
  add_model(ols_spec) |> 
  add_recipe(lm_rec)

# Fit workflows ----

# set up parallel network sockets
cores <- parallel::detectCores(logical = FALSE) - 1
c1 <- makePSOCKcluster(cores)
registerDoParallel(c1)

# fit workflow
ols_fit <- ols_wflow |> 
  fit_resamples(
    airbnb_folds, 
    control = keep_wflow_res,
    metrics = my_metrics
  )

# reset to sequential processing
stopCluster(c1)
registerDoSEQ()
rm(c1)

# Write out results ----
save(ols_fit, file = here("attempt_1/results/ols_fit.rda"))
