# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: fit logistic regression models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls
load(here("attempt_1/data_splits/keep_wflow_res.rda"))

# linear recipe
load(here("attempt_1/recipes/lm_rec.rda"))

# Model specification ----
log_spec <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

# Define workflow ----
log_wflow <- workflow() |> 
  add_model(log_spec) |> 
  add_recipe(lm_rec)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
log_fit <- log_wflow |> 
  fit_resamples(
    airbnb_folds, 
    control = keep_wflow_res
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(log_fit, file = here("attempt_1/results/log_fit.rda"))
