# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 4: model comparison and selection of final models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# fitted models
list.files(
  path = here("attempt_1/results/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# package as workflow set
tune_results <- as_workflow_set(
  nb = nb_fit,
  log = log_fit,
  en = en_tuned,
  # knn_lm = knn_lm_tuned,
  # knn_tree = knn_tree_tuned,
  # rf = rf_tuned,
  bt = bt_tuned,
  # svm_poly = svm_poly_tuned,
  # svm_rbf = svm_rbf_tuned,
  mars = mars_tuned,
  nn = nn_tuned
)

# Compare performance metrics ----

# roc_auc is the final performance metric

all_models <- tune_results |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  arrange(-mean)

all_models |> view()

tune_results |>
  autoplot(metric = "roc_auc", select_best = TRUE)

# boosted tree is the best by far so far
# nb, linear models are pretty poor as well

# Analyze tuning values ----

# function to extract all mae metrics
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

## nb ----

# try this as initial testing
final_nb <- nb_fit |> 
  extract_workflow()

# save workflow
save(final_nb, file = here("attempt_1/submissions/workflows/final_nb.rda"))

## log ----

# try this as initial testing
final_log <- log_fit |> 
  extract_workflow()

# save workflow
save(final_log, file = here("attempt_1/submissions/workflows/final_log.rda"))

## en ----
# en_tuned |>
#   autoplot(metric = "roc_auc")
# 
# en_tuned |>
#   roc_auc_metrics()

# winning model (also not great)
final_en <- en_tuned |> 
  extract_workflow() |> 
  finalize_workflow(get_hyperparams(en_tuned, 1, c(penalty, mixture)))

# save workflow
save(final_en, file = here("attempt_1/submissions/workflows/final_en.rda"))

## linear knn ----
# knn_lm_tuned |> 
#   autoplot(metric = "mae")
# 
# knn_lm_tuned |> 
#   mae_metrics()

# top models
final_knn_lm <- c(1, 2, 3, 4, 5) |> 
  map(
    \(x) knn_lm_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(knn_lm_tuned, x, neighbors))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_knn_lm, file = here("attempt_1/submissions/workflows/final_knn_lm.rda"))

## tree knn ----
# knn_tree_tuned |> 
#   autoplot(metric = "mae")
# 
# knn_tree_tuned |> 
#   mae_metrics()

# top models
final_knn_tree <- c(1, 2, 3) |> 
  map(
    \(x) knn_tree_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(knn_tree_tuned, x, neighbors))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_knn_tree, file = here("attempt_1/submissions/workflows/final_knn_tree.rda"))

## rf ----


## bt ----
# bt_tuned |>
#   autoplot(metric = "roc_auc")
# 
# bt_tuned |>
#   roc_auc_metrics()

# top models
final_bt <- c(1, 2, 3, 4, 5) |> 
  map(
    \(x) bt_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(bt_tuned, x, c(mtry, trees, min_n, learn_rate)))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_bt, file = here("attempt_1/submissions/workflows/final_bt.rda"))

## poly svm ----


## rbf svm ----
# svm_rbf_tuned |> 
#   autoplot(metric = "mae")
# 
# svm_rbf_tuned |> 
#   mae_metrics()

# winning model, no others close
final_svm_rbf <- svm_rbf_tuned |> 
  extract_workflow() |> 
  finalize_workflow(get_hyperparams(svm_rbf_tuned, 1, c(cost, rbf_sigma)))

# save workflow
save(final_svm_rbf, file = here("attempt_1/submissions/workflows/final_svm_rbf.rda"))

## mars ----
# mars_tuned |>
#   autoplot(metric = "roc_auc")
# 
# mars_tuned |>
#   roc_auc_metrics()

# top models
final_mars <- c(1, 2, 3) |> 
  map(
    \(x) mars_tuned |> 
      extract_workflow() |> 
      finalize_workflow(get_hyperparams(mars_tuned, x, c(num_terms, prod_degree)))
  ) |> 
  as_tibble_col(column_name = "workflow")

# save workflows
save(final_mars, file = here("attempt_1/submissions/workflows/final_mars.rda"))

## nn ----
# nn_tuned |>
#   autoplot(metric = "roc_auc")
# 
# nn_tuned |>
#   roc_auc_metrics()

# winning model (pretty good, but could improve tuning)
final_nn <- nn_tuned |> 
  extract_workflow() |> 
  finalize_workflow(get_hyperparams(nn_tuned, 1, c(hidden_units, penalty)))

# save workflow
save(final_nn, file = here("attempt_1/submissions/workflows/final_nn.rda"))
