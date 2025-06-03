# Classification Prediction Problem ----
# Stat 301-3
# Final model 1
# Step 4: model comparison and selection of final models

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(bonsai)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# boosted tree models
load(here("final_submissions/model_1/results/btl_tuned.rda"))

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
save(final_btl, file = here("final_submissions/model_1/results/final_btl.rda"))
