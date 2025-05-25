# Classification Prediction Problem ----
# Stat 301-3
# Attempt 2
# Step 4: model comparison and selection of final models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# fitted models
list.files(
  path = here("attempt_2/results/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# package as workflow set
tune_results <- as_workflow_set(
  btx = btx_tuned,
  rf = rf_tuned,
  btl = btl_tuned
)

# Compare performance metrics ----

# roc_auc is the final performance metric

# all_models <- tune_results |>
#   collect_metrics() |>
#   filter(.metric == "roc_auc") |>
#   arrange(-mean)
# 
# all_models |> view()
# 
# tune_results |>
#   autoplot(metric = "roc_auc", select_best = TRUE, std_errs = 1)

# lightgbm boosted trees are the best by a significant amount
# random forest appears to perform quite poorly here

# Analyze tuning values ----

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

## btx ----
# btx_tuned |>
#   autoplot(metric = "roc_auc")
# 
# btx_tuned |>
#   roc_auc_metrics()

# appears to be maximum with learn rate, use smallest min_n
# could try larger values of trees, specific tuning for mtry

# top models
final_btx <- c(1, 2, 3, 4) |> 
  map(
    \(x) btx_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(btx_tuned, x, c(mtry, trees, min_n, learn_rate)))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_btx, file = here("attempt_2/submissions/workflows/final_btx.rda"))

## rf ----
# rf_tuned |>
#   autoplot(metric = "roc_auc")
# 
# rf_tuned |>
#   roc_auc_metrics()

# smaller min_n gives best performance, appears to have some peak at moderate mtry

# top models
final_rf <- c(1, 2) |> 
  map(
    \(x) rf_tuned |> 
      extract_workflow() |> 
      finalize_workflow(c(get_hyperparams(rf_tuned, x, c(mtry, min_n)), trees = 500))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_rf, file = here("attempt_2/submissions/workflows/final_rf.rda"))

## btl ----
# btl_tuned |>
#   autoplot(metric = "roc_auc")
# 
# btl_tuned |>
#   roc_auc_metrics() |> view()

# appears to be maximum with learn rate, use smaller min_n values
# could try larger values of trees, specific tuning for mtry
# top 10 models are all within 1 standard error

# top models
final_btl <- c(1:10) |>
  map(
    \(x) btl_tuned |>
      extract_workflow() |>
      finalize_workflow(get_hyperparams(btl_tuned, x, c(mtry, trees, min_n, learn_rate)))
  ) |>
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_btl, file = here("attempt_2/submissions/workflows/final_btl.rda"))
