# Regression Prediction Problem ----
# Stat 301-3
# Final model 2
# Step 5: make final predictions on test set

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
load(here("attempt_3/submissions/fitted/bt_final_fits.rda"))

# Add new column ----
test_clean <- test_clean |> 
  add_count(host_about, host_since) |> 
  rename(host_match = n)

# Compute predictions ----

# function to compute predictions
compute_preds <- function(model) {
  test_clean |> 
    select(id) |> 
    bind_cols(model |> predict(test_clean, type = "prob")) |> 
    select(id, predicted = .pred_1)
}

# compute predictions for the submitted model
model_2_preds <- bt_final_fits |> 
  pull(fit) |> 
  pluck(1) |> 
  compute_preds()

# write out results
write_csv(model_2_preds, file = here("final_submissions/model_2/results/model_2_preds.csv"))
