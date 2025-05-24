# Regression Prediction Problem ----
# Stat 301-3
# Attempt 2
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

# fitted workflows
list.files(
  path = here("attempt_2/submissions/fitted/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# Compute predictions ----

# function to compute predictions
compute_preds <- function(model) {
  test_clean |> 
    select(id) |> 
    bind_cols(model |> predict(test_clean, type = "prob")) |> 
    select(id, predicted = .pred_1)
}

# all predictions commented out to avoid accidentally changing values

## btx ----
# btx_preds <- btx_fits |>
#   mutate(
#     preds = map(btx_fits$fit, compute_preds)
#   )
# 
# # write out results
# for (ind in c(1, 2, 3, 4)) {
#   write_csv(btx_preds$preds[[ind]], file = here(paste0(
#     "attempt_2/submissions/preds/btx_preds_", as.character(ind), ".csv"
#   )))
# }

## rf ----
# rf_preds <- rf_fits |>
#   mutate(
#     preds = map(rf_fits$fit, compute_preds)
#   )
# 
# # write out results
# for (ind in c(1, 2, 3, 4)) {
#   write_csv(rf_preds$preds[[ind]], file = here(paste0(
#     "attempt_2/submissions/preds/rf_preds_", as.character(ind), ".csv"
#   )))
# }

## btl ----
# btl_preds <- btl_fits |>
#   mutate(
#     preds = map(btl_fits$fit, compute_preds)
#   )
# 
# # write out results
# for (ind in c(1, 2, 3)) {
#   write_csv(btl_preds$preds[[ind]], file = here(paste0(
#     "attempt_2/submissions/preds/btl_preds_", as.character(ind), ".csv"
#   )))
# }
