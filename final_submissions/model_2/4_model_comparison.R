# Regression Prediction Problem ----
# Stat 301-3
# Final model 2
# Step 4: model comparison and selection/fitting of final models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# full training data
load(here("final_submissions/model_2/data_splits/airbnb.rda"))

# boosted tree models
load(here("final_submissions/model_2/results/bt_tuned.rda"))

# Select final models ----

# function to extract all roc_auc metrics
roc_auc_metrics <- function(result) {
  result |> 
    collect_metrics() |> 
    filter(.metric == "roc_auc") |> 
    arrange(-mean)
}

# function to get n-th best hyperparameters
get_hyperparams <- function(result, n, params) {
  result |> 
    roc_auc_metrics() |> 
    filter(row_number() == {{ n }}) |> 
    select({{ params }})
}

# thorough analysis to arrive at these results is in the attempt_3 folder

# top model workflows
final_bt <- c(1:25) |> 
  map(
    \(x) bt_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(bt_tuned, x, c(mtry, trees, min_n, tree_depth, learn_rate)))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_bt, file = here("final_submissions/model_2/results/final_bt.rda"))

# Fit final workflows ----

# set seed (to run separately)
set.seed(3189)

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# fit workflows
bt_final_fits <- final_bt |>
  mutate(
    fit = map(final_bt$workflow, \(x) fit(x, airbnb))
  )

# write out results
save(bt_final_fits, file = here("final_submissions/model_2/results/bt_final_fits.rda"))
