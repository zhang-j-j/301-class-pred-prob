# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune mars models

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

# tree recipe
load(here("attempt_3/recipes/lm_rec.rda"))

# Model specification ----
mars_spec <- mars(
  num_terms = tune(),
  prod_degree = tune()
) |> 
  set_engine("earth") |> 
  set_mode("classification")

# Define workflow ----
mars_wflow <- workflow() |> 
  add_model(mars_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
mars_params <- extract_parameter_set_dials(mars_wflow) |> 
  update(
    num_terms = num_terms(c(1, 70)),
    prod_degree = prod_degree()
  )

# build tuning grid
mars_grid <- grid_regular(
  mars_params,
  levels = c(num_terms = 8, prod_degree = 2)
)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
mars_tuned <- mars_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = mars_grid,
    control = control_stack_grid()
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(mars_tuned, file = here("attempt_3/results/mars_tuned.rda"))
