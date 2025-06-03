# Regression Prediction Problem ----
# Stat 301-3
# Final model 1
# Step 6: make final predictions on test set

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# testing data
load(here("data/test_clean.rda"))

# fitted bt workflows
load(here("final_submissions/model_1/results/btl_fits.rda"))

# Compute predictions ----

# function to compute predictions
compute_preds <- function(model) {
  test_clean |> 
    select(id) |> 
    bind_cols(model |> predict(test_clean, type = "prob")) |> 
    select(id, predicted = .pred_1)
}

# compute predictions for the submitted model
model_1_preds <- btl_fits |> 
  pull(fit) |> 
  pluck(2) |> 
  compute_preds()

# write out results
write_csv(model_1_preds, file = here("final_submissions/model_1/results/model_1_preds.csv"))
