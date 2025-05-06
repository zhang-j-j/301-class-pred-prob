# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 1: data splitting/folding, general setup

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(2149)

# Load data ----
load(here("data/train_clean.rda"))

# for this attempt, just use this as the entire training set
# could consider also doing an initial split in later attempts and comparing

# Data prep ----

# use a log-transformation in this attempt
airbnb_train <- train_clean |> 
  mutate(price_log10 = log10(price))

# skimr::skim_without_charts(airbnb)

# Fold data ----

# use 10 folds, 3 repeats, stratify by target variable
airbnb_folds <- airbnb_train |> 
  vfold_cv(
    v = 10,
    repeats = 3,
    strata = price_log10
  )

# Set up controls ----
keep_wflow_res <- control_resamples(save_workflow = TRUE)
keep_wflow_grid <- control_grid(save_workflow = TRUE)
my_metrics <- metric_set(mae, rmse, rsq)

# Save objects ----
save(airbnb_train, file = here("attempt_1/data_splits/airbnb_train.rda"))
save(airbnb_folds, file = here("attempt_1/data_splits/airbnb_folds.rda"))
save(keep_wflow_res, file = here("attempt_1/data_splits/keep_wflow_res.rda"))
save(keep_wflow_grid, file = here("attempt_1/data_splits/keep_wflow_grid.rda"))
save(my_metrics, file = here('attempt_1/data_splits/my_metrics.rda'))
