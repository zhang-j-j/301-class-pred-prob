# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune elastic net models

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
en_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) |> 
  set_engine("glmnet") |> 
  set_mode("classification")

# Define workflow ----
en_wflow <- workflow() |> 
  add_model(en_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
en_params <- extract_parameter_set_dials(en_wflow) |> 
  update(
    penalty = penalty(c(-4, -1)),
    mixture = mixture()
  )

# build tuning grid
en_grid <- grid_regular(
  en_params,
  levels = c(penalty = 7, mixture = 6)
)

# Fit workflows ----

# runs much faster without parallel processing
# # set up parallel processing
# cores <- availableCores() - 1
# plan(multisession, workers = cores)

# fit workflow
en_tuned <- en_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = en_grid,
    control = control_stack_grid()
  )

# # reset to sequential processing
# plan(sequential)

# Write out results ----
save(en_tuned, file = here("attempt_3/results/en_tuned.rda"))
