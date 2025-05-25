# Classification Prediction Problem ----
# Stat 301-3
# Attempt 2
# Step 5: fit selected models for submissions

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# training data
load(here("attempt_2/data_splits/airbnb_train.rda"))

# final workflows
list.files(
  path = here("attempt_2/submissions/workflows/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# Fit final workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# all fits commented out to avoid refitting/changing models

## btx ----
# # set seed (to run separately)
# set.seed(201)
# 
# btx_fits <- final_btx |>
#   mutate(
#     fit = map(final_btx$workflow, \(x) fit(x, airbnb_train))
#   )
# 
# # write out results
# save(btx_fits, file = here("attempt_2/submissions/fitted/btx_fits.rda"))

## rf ----
# # set seed (to run separately)
# set.seed(25)
# 
# rf_fits <- final_rf |>
#   mutate(
#     fit = map(final_rf$workflow, \(x) fit(x, airbnb_train))
#   )
# 
# # write out results
# save(rf_fits, file = here("attempt_2/submissions/fitted/rf_fits.rda"))

## btl ----
# # set seed (to run separately)
# set.seed(408)
# 
# btl_fits <- final_btl |>
#   mutate(
#     fit = map(final_btl$workflow, \(x) fit(x, airbnb_train))
#   )
# 
# # write out results
# save(btl_fits, file = here("attempt_2/submissions/fitted/btl_fits.rda"))
