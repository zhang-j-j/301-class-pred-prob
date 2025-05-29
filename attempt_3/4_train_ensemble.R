# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 4: train ensemble models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
library(future)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# candidate models
list.files(
  path = here("attempt_3/results/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# package as workflow set
tune_results <- as_workflow_set(
  log = log_fit,
  bt = bt_tuned,
  en = en_tuned,
  knn = knn_tuned,
  mars = mars_tuned,
  nn = nn_tuned,
  svm = svm_tuned
)

# workflow set without bt models
no_bt_results <- as_workflow_set(
  log = log_fit,
  en = en_tuned,
  knn = knn_tuned,
  mars = mars_tuned,
  nn = nn_tuned,
  svm = svm_tuned
)

# Ensemble 1 ----

# directly use all of the fitted models as candidate models
ens_1_data <- stacks() |> 
  add_candidates(tune_results)

# # check the stack
# ens_1_data |> as_tibble()

## fit stack ----

# set up parallel processing (don't use too many cores, leads to crashing)
plan(multisession, workers = 5)

# set seed (to run separately)
set.seed(215)

# blend predictions
ens_1_models <- ens_1_data |> 
  blend_predictions(
    penalty = c(10^(-6:-1), 0.5, 1, 1.5, 2), 
    metric = metric_set(mae)
  )

# fit members
ens_1_fit <- ens_1_models |> 
  fit_members()

# reset to sequential processing
plan(sequential)

## save model stacks ----
save(ens_1_models, file = here("attempt_3/results/ensemble/ens_1_models.rda"))
save(ens_1_fit, file = here("attempt_3/results/ensemble/ens_1_fit.rda"))

# Ensemble 2 ----

# don't include boosted trees
ens_2_data <- stacks() |> 
  add_candidates(no_bt_results)

# check the stack
ens_2_data |> as_tibble() |> 
  skimr::skim_without_charts()

## fit stack ----

# set up parallel processing (don't use too many cores, leads to crashing)
plan(multisession, workers = 5)

# set seed (to run separately)
set.seed(229)

# blend predictions
ens_2_models <- ens_2_data |> 
  blend_predictions(
    penalty = c(10^(-6:-1), 0.5, 1, 1.5, 2),
    metric = metric_set(mae)
  )

# fit members
ens_2_fit <- ens_2_models |> 
  fit_members()

# reset to sequential processing
plan(sequential)

## save model stacks ----
save(ens_2_models, file = here("attempt_3/results/ensemble/ens_2_models.rda"))
save(ens_2_fit, file = here("attempt_3/results/ensemble/ens_2_fit.rda"))
