# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
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

# training data
load(here("attempt_3/data_splits/airbnb_train.rda"))

# fitted models
list.files(
  path = here("attempt_3/results/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# package as workflow set
tune_results <- as_workflow_set(
  log = log_fit,
  en = en_tuned,
  knn = knn_tuned,
  bt = bt_tuned,
  svm = svm_tuned,
  mars = mars_tuned,
  nn = nn_tuned
)

# Compare performance metrics ----

# roc_auc is the final performance metric
# at this point, only really care about the individual bt models

# all_models <- tune_results |>
#   collect_metrics() |>
#   filter(.metric == "roc_auc") |>
#   arrange(-mean)
# 
# all_models |> view()
# 
# tune_results |>
#   autoplot(metric = "roc_auc", select_best = TRUE)

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

## bt ----
# bt_tuned |>
#   autoplot(metric = "roc_auc")
# 
# bt_tuned |>
#   roc_auc_metrics() |> view()

# these might actually be slightly worse than previous attempts (less training data
# used so far), but generally similar results
# a lot of them converge at a very good performance (top 25 within 1 standard error)

# top models
final_bt <- c(1:25) |> 
  map(
    \(x) bt_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(bt_tuned, x, c(mtry, trees, min_n, tree_depth, learn_rate)))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_bt, file = here("attempt_3/submissions/workflows/final_bt.rda"))

# Fit bt models ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# set seed (to run separately)
set.seed(30)

# fit workflows
bt_fits <- final_bt |>
  mutate(
    fit = map(final_bt$workflow, \(x) fit(x, airbnb_train))
  )

# write out results
save(bt_fits, file = here("attempt_3/submissions/fitted/bt_fits.rda"))
