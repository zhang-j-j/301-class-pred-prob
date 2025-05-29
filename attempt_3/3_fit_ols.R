# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: fit ols models

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

# controls and metrics
load(here("attempt_3/data_splits/my_metrics.rda"))

# tree recipe
load(here("attempt_3/recipes/lm_rec.rda"))

# Model specification ----
ols_spec <- linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression")

# Define workflow ----
ols_wflow <- workflow() |> 
  add_model(ols_spec) |> 
  add_recipe(lm_rec)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
ols_fit <- ols_wflow |> 
  fit_resamples(
    airbnb_folds, 
    control = control_stack_resamples(),
    metrics = my_metrics
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(ols_fit, file = here("attempt_3/results/ols_fit.rda"))
