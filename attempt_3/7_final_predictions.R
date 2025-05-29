# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 7: make final predictions on test set

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# testing data
load(here("data/test_clean.rda"))
## remember to add the column

# fitted bt workflows
load(here("attempt_3/submissions/fitted/bt_final_fits.rda"))

# fitted ensemble models
load(here("attempt_3/results/ensemble/ens_1_fit.rda"))
load(here("attempt_3/results/ensemble/ens_2_fit.rda"))

# Compute predictions ----

# function to compute predictions (try rounding to nearest whole number)
compute_preds <- function(model) {
  test_clean |> 
    select(id) |> 
    bind_cols(model |> predict(test_clean)) |> 
    mutate(predicted = round(10 ^ .pred)) |> 
    select(-.pred)
}

# all predictions commented out to avoid accidentally changing values

# bt ----
bt_preds <- bt_final_fits |>
  mutate(
    preds = map(bt_final_fits$fit, compute_preds)
  )

# write out results
for (ind in c(1:10)) {
  write_csv(bt_preds$preds[[ind]], file = here(paste0(
    "attempt_3/submissions/preds/bt_preds_", as.character(ind), ".csv"
  )))
}

## ens 1 ----
ens_1_members <- test_clean |> 
  select(id) |> 
  bind_cols(ens_1_fit |> predict(test_clean, members = TRUE))

# selected member models
ens_1_names <- c(30, 39, 89, 318, 31, 43, 92, 21) |> 
  map(\(x) if_else(
    x > 100, paste0("bt_1_", as.character(x)),
    paste0("bt_1_0", as.character(x))
  )) |> 
  as.character()

# write out results
for (name in c(".pred", ens_1_names)) {
  ens_1_members |> 
    select(id, predicted = {{ name }}) |> 
    mutate(predicted = round(10 ^ predicted)) |> 
    write_csv(file = here(paste0(
      "attempt_3/submissions/preds/ens_preds_", as.character(name), ".csv"
    )))
}

## ens 2 ----
ens_2_preds <- ens_2_fit |> 
  compute_preds()

# write out results
write_csv(ens_2_preds, file = here("attempt_3/submissions/preds/ens_2_preds.csv"))
