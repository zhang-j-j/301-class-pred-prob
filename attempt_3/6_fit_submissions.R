# Regression Prediction Problem ----
# Stat 301-3
# Attempt 6
# Step 6: fit selected models for submissions

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(3189)

# Load objects ----

# training data
load(here("attempt_3/data_splits/airbnb.rda"))

# bt workflows
load(here("attempt_3/submissions/workflows/final_bt.rda"))

# doesn't appear possible to retrain the ensemble models, so use them as-is

# Fit final workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# bt ----
bt_final_fits <- final_bt |>
  mutate(
    fit = map(final_bt$workflow, \(x) fit(x, airbnb))
  )

# write out results
save(bt_final_fits, file = here("attempt_3/submissions/fitted/bt_final_fits.rda"))
