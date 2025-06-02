# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 5: assess ensemble model and individual final models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
library(bonsai)

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

# 10 member models, all boosted trees
ens_1_models |> autoplot(type = "weights")

ens_1_models |> autoplot()

ens_1_models |> autoplot(type = "members")

## ens_2 ----

# 15 member models, mostly svm but some mars, 1 nn and 1 knn
ens_2_models |> autoplot(type = "weights")

ens_2_models |> autoplot()

ens_2_models |> autoplot(type = "members")

# this one seems to perform slightly worse than the other ensemble

# Compute predictions ----

# function to compute predictions
compute_preds <- function(model) {
  airbnb_test |> 
    select(host_is_superhost) |> 
    bind_cols(model |> predict(airbnb_test, type = "prob"))
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
  select(host_is_superhost) |> 
  bind_cols(ens_1_fit |> predict(airbnb_test, type = "prob", members = TRUE))

ens_2_members <- airbnb_test |> 
  select(host_is_superhost) |> 
  bind_cols(ens_2_fit |> predict(airbnb_test, type = "prob", members = TRUE))

# Compute metrics ----

# function to compute metrics
compute_metrics <- function(preds) {
  preds |> 
    summarize(
      roc_auc(preds, host_is_superhost, .pred_1)
    )
}

## bt ----
bt_rank <- c(1:25) |> 
  map(
    \(x) bt_preds |> 
      pull(preds) |> 
      pluck(x) |>
      compute_metrics()
  ) |> 
  bind_rows() |> 
  mutate(id = as.character(row_number())) |> 
  arrange(-.estimate)

# bt ranking: 1, 14, 10, 3, 13, 2, 22, 4, 18, 17, 16, 25, 7, 11, 19, 15, 12, 5, 23, 24
# 8, 6, 21, 9, 20

## ens ----
full_rank <- bind_rows(
  bt_rank,
  ens_1_preds |> compute_metrics() |> mutate(id = "ens1"),
  ens_2_preds |> compute_metrics() |> mutate(id = "ens2")
) |> 
  arrange(-.estimate)

full_rank |> view()

# notes: the best individual boosted tree is best, first ensemble is near the top
# the second ensemble is pretty bad
# these are all relatively reasonable results

## ens members ----
ens_1_members |> 
  select(host_is_superhost, contains(".pred_1")) |> 
  summarize(
    across(
      -host_is_superhost,
      \(x) roc_auc(ens_1_members, host_is_superhost, x) |> pull(.estimate)
    )
  ) |> 
  pivot_longer(cols = everything()) |> 
  arrange(-value)

# try all of the member models individually

ens_2_members |> 
  summarize(
    across(
      -host_is_superhost, 
      \(x) roc_auc(ens_2_members, host_is_superhost, x) |> pull(.estimate)
    )
  ) |> 
  pivot_longer(cols = everything()) |> 
  arrange(-value)

# try the ensemble, but don't bother with any members
