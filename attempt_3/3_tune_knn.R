# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 3: tune knn models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
library(future)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(1256)

# Load objects ----

# resamples
load(here("attempt_3/data_splits/airbnb_folds.rda"))

# ljnear recipe
load(here("attempt_3/recipes/lm_rec.rda"))

# Model specification ----
knn_spec <- nearest_neighbor(
  neighbors = tune()
) |> 
  set_engine("kknn") |> 
  set_mode("classification")

# Define workflow ----
knn_wflow <- workflow() |> 
  add_model(knn_spec) |> 
  add_recipe(lm_rec)

# Hyperparameter tuning values ----

# change hyperparameter ranges
knn_params <- extract_parameter_set_dials(knn_wflow) |> 
  update(
    neighbors = neighbors(c(11, 25))
  )

# build tuning grid
knn_grid <- grid_regular(
  knn_params,
  levels = c(neighbors = 15)
)

# Fit workflows ----

# runs much faster without parallel processing
# # set up parallel processing
# cores <- availableCores() - 1
# plan(multisession, workers = cores)

# fit workflow
knn_tuned <- knn_wflow |> 
  tune_grid(
    airbnb_folds, 
    grid = knn_grid,
    control = control_stack_grid()
  )

# # reset to sequential processing
# plan(sequential)

# Write out results ----
save(knn_tuned, file = here("attempt_3/results/knn_tuned.rda"))
