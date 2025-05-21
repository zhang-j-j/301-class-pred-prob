# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
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
  path = here("attempt_1/submissions/fitted/"),
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

## nb ----
nb_preds <- nb_fit |> 
  compute_preds()

# write out results
write_csv(nb_preds, file = here("attempt_1/submissions/preds/nb_preds.csv"))

## log ----
log_preds <- log_fit |> 
  compute_preds()

# write out results
write_csv(log_preds, file = here("attempt_1/submissions/preds/log_preds.csv"))

## en ----
en_preds <- en_fit |> 
  compute_preds()

# write out results
write_csv(en_preds, file = here("attempt_1/submissions/preds/en_preds.csv"))

## linear knn ----
knn_lm_preds <- knn_lm_fits |>
  mutate(
    preds = map(knn_lm_fits$fit, compute_preds)
  )

# write out results
for (ind in c(1, 2, 3)) {
  write_csv(knn_lm_preds$preds[[ind]], file = here(paste0(
    "attempt_1/submissions/preds/knn_lm_preds_", as.character(ind), ".csv"
  )))
}

## tree knn ----
knn_tree_preds <- knn_tree_fits |>
  mutate(
    preds = map(knn_tree_fits$fit, compute_preds)
  )

# write out results
for (ind in c(1, 2, 3, 4, 5)) {
  write_csv(knn_tree_preds$preds[[ind]], file = here(paste0(
    "attempt_1/submissions/preds/knn_tree_preds_", as.character(ind), ".csv"
  )))
}

## rf ----

rf_preds <- rf_fits |>
  mutate(
    preds = map(rf_fits$fit, compute_preds)
  )

# write out results
for (ind in c(1, 2, 3, 4)) {
  write_csv(rf_preds$preds[[ind]], file = here(paste0(
    "attempt_1/submissions/preds/rf_preds_", as.character(ind), ".csv"
  )))
}

## bt ----
bt_preds <- bt_fits |> 
  mutate(
    preds = map(bt_fits$fit, compute_preds)
  )

# write out results
for (ind in c(1, 2, 3, 4, 5)) {
  write_csv(bt_preds$preds[[ind]], file = here(paste0(
    "attempt_1/submissions/preds/bt_preds_", as.character(ind), ".csv"
  )))
}

## rbf svm ----
svm_rbf_preds <- svm_rbf_fit |>
  compute_preds()

# write out results
write_csv(svm_rbf_preds, file = here("attempt_1/submissions/preds/svm_rbf_preds.csv"))

## mars ----
mars_preds <- mars_fits |> 
  mutate(
    preds = map(mars_fits$fit, compute_preds)
  )

# write out results
for (ind in c(1, 2, 3)) {
  write_csv(mars_preds$preds[[ind]], file = here(paste0(
    "attempt_1/submissions/preds/mars_preds_", as.character(ind), ".csv"
  )))
}

## nn ----
nn_preds <- nn_fit |> 
  compute_preds()

# write out results
write_csv(nn_preds, file = here("attempt_1/submissions/preds/nn_preds.csv"))
