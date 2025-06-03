# Classification Prediction Problem ----
# Stat 301-3
# Final model 1
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
load(here("final_submissions/model_1/data_splits/airbnb_train.rda"))

# final bt workflows
load(here("final_submissions/model_1/results/final_btl.rda"))

# Fit final workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

## btl ----
# set seed (to run separately)
set.seed(408)

btl_fits <- final_btl |>
  mutate(
    fit = map(final_btl$workflow, \(x) fit(x, airbnb_train))
  )

# write out results
save(btl_fits, file = here("final_submissions/model_1/results/btl_fits.rda"))
