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

# supplemental data
load(here("data/train_clean.rda"))

# fitted bt workflows
load(here("attempt_3/submissions/fitted/bt_final_fits.rda"))

# fitted ensemble models
load(here("attempt_3/results/ensemble/ens_1_fit.rda"))
load(here("attempt_3/results/ensemble/ens_2_fit.rda"))

# Add new column ----

# combine and recount column (total count)
test_clean_1 <- bind_rows(test_clean, train_clean) |> 
  add_count(host_about, host_since) |> 
  rename(host_match = n) |> 
  filter(is.na(host_is_superhost)) |> 
  select(-host_is_superhost)

# count within testing set only
test_clean_2 <- test_clean |> 
  add_count(host_about, host_since) |> 
  rename(host_match = n)

# appears that test_clean_2 performs better based on first few bt submissions, so
# only use that for the ensembles

# Compute predictions ----

# function to compute predictions
compute_preds <- function(model, test) {
  test |> 
    select(id) |> 
    bind_cols(model |> predict(test, type = "prob")) |> 
    select(id, predicted = .pred_1)
}

# all predictions commented out to avoid accidentally changing values

# bt ----
bt_preds <- bt_final_fits |>
  mutate(
    preds1 = map(bt_final_fits$fit, \(x) compute_preds(x, test_clean_1)),
    preds2 = map(bt_final_fits$fit, \(x) compute_preds(x, test_clean_2))
  )

# # write out results
# for (ind in c(1:25)) {
#   write_csv(bt_preds$preds1[[ind]], file = here(paste0(
#     "attempt_3/submissions/preds/bt_preds1_", as.character(ind), ".csv"
#   )))
#   write_csv(bt_preds$preds2[[ind]], file = here(paste0(
#     "attempt_3/submissions/preds/bt_preds2_", as.character(ind), ".csv"
#   )))
# }

## ens 1 ----
ens_1_members <- test_clean |> 
  select(id) |> 
  bind_cols(ens_1_fit |> predict(test_clean_2, type = "prob", members = TRUE))

# selected member models
ens_1_names <- c(139, 286, 142, 175, 245, 282, 247, 241, 281, 130) |> 
  map(\(x) paste0(".pred_1_bt_1_", as.character(x))) |> 
  as.character()

# # write out results
# for (name in c(".pred_1", ens_1_names)) {
#   ens_1_members |> 
#     select(id, predicted = {{ name }}) |> 
#     write_csv(file = here(paste0(
#       "attempt_3/submissions/preds/ens_preds_", substring(as.character(name), 9), ".csv"
#     )))
# }

## ens 2 ----
# ens_2_preds <- ens_2_fit |> 
#   compute_preds(test_clean_2)
# 
# # write out results
# write_csv(ens_2_preds, file = here("attempt_3/submissions/preds/ens_2_preds.csv"))
