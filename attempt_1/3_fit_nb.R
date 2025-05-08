# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 3: fit naive bayes models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(discrim)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# resamples
load(here("attempt_1/data_splits/airbnb_folds.rda"))

# controls
load(here("attempt_1/data_splits/keep_wflow_res.rda"))

# naive bayes recipe
load(here("attempt_1/recipes/nb_rec.rda"))

# Model specification ----
nb_spec <- naive_Bayes() |> 
  set_engine("klaR") |> 
  set_mode("classification")

# Define workflow ----
nb_wflow <- workflow() |> 
  add_model(nb_spec) |> 
  add_recipe(nb_rec)

# Fit workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflow
nb_fit <- nb_wflow |> 
  fit_resamples(
    airbnb_folds, 
    control = keep_wflow_res
  )

# reset to sequential processing
plan(sequential)

# Write out results ----
save(nb_fit, file = here("attempt_1/results/nb_fit.rda"))
