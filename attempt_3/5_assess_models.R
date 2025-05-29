# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 5: assess ensemble model and individual final models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# testing data
load(here("attempt_3/data_splits/airbnb_test.rda"))

# boosted tree models
load(here("attempt_3/submissions/fitted/bt_fits.rda"))

# ensemble models
list.files(
  path = here("attempt_3/results/ensemble/"),
  pattern = "ens_",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# Analyze ensembles ----

## ens_1 ----

# 21 member models, mostly boosted trees
ens_1_models |> autoplot(type = "weights")

ens_1_models |> autoplot()

ens_1_models |> autoplot(type = "members")

## ens_2 ----

# 16 member models, good variety between mars, nn, knn, svm
ens_2_models |> autoplot(type = "weights")

ens_2_models |> autoplot()

ens_2_models |> autoplot(type = "members")

# this one seems to perform much worse than the other ensemble

# Compute predictions ----

# function to compute predictions
compute_preds <- function(model) {
  airbnb_test |> 
    select(transformed = price_log10) |> 
    bind_cols(model |> predict(airbnb_test)) |> 
    mutate(
      original = 10 ^ transformed,
      .pred_orig = 10 ^ .pred
    )
}

# bt ----
bt_preds <- bt_fits |>
  mutate(
    preds = map(bt_fits$fit, compute_preds)
  )

## ens ----
ens_1_preds <- ens_1_fit |> 
  compute_preds()

ens_2_preds <- ens_2_fit |> 
  compute_preds()

## ens members ----
ens_1_members <- airbnb_test |> 
  select(price_log10) |> 
  bind_cols(ens_1_fit |> predict(airbnb_test, members = TRUE))

ens_2_members <- airbnb_test |> 
  select(price_log10) |> 
  bind_cols(ens_2_fit |> predict(airbnb_test, members = TRUE))

# Compute metrics ----

# function to compute metrics
compute_metrics <- function(preds) {
  preds |> 
    summarize(
      mae_log = mae(preds, truth = transformed, estimate = .pred) |> pull(.estimate),
      mae_orig = mae(preds, truth = original, estimate = .pred_orig) |> pull(.estimate),
    )
}

## bt ----
bt_rank <- c(1:10) |> 
  map(
    \(x) bt_preds |> 
      pull(preds) |> 
      pluck(x) |>
      compute_metrics()
  ) |> 
  bind_rows() |> 
  mutate(id = as.character(row_number()))

# bt ranking (original scale): 2, 7, 1, 8, 3, 4, 5, 6, 9, 10

## ens ----
full_rank <- bind_rows(
  bt_rank,
  ens_1_preds |> compute_metrics() |> mutate(id = "ens1"),
  ens_2_preds |> compute_metrics() |> mutate(id = "ens2")
) |> 
  arrange(mae_orig)

full_rank |> view()

# the first ensemble is slightly better than all of the boosted trees
# the second ensemble is pretty bad
# these metrics seem suspiciously low, but keep going and see what happens

## ens members ----
ens_1_members |> 
  summarize(
    across(
      -price_log10,
      \(x) mae(ens_1_members, truth = price_log10, estimate = x) |> pull(.estimate)
    )
  ) |> 
  pivot_longer(cols = everything()) |> 
  arrange(value) |> 
  print(n = 22)

# try the following: ensemble, 30, 39, 89, 318, 31, 43, 92, 21

ens_2_members |> 
  summarize(
    across(
      -price_log10, 
      \(x) mae(ens_2_members, truth = price_log10, estimate = x) |> pull(.estimate)
    )
  ) |> 
  pivot_longer(cols = everything()) |> 
  arrange(value)

# try the ensemble, but don't bother with any members

# difficult to compute these metrics on the original scale, so stick with the log-scale
# interesting that one of the boosted trees outperforms the ensemble, but wasn't 
# selected in the prior step
