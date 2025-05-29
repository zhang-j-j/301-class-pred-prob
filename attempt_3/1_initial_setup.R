# Regression Prediction Problem ----
# Stat 301-3
# Attempt 3
# Step 1: data splitting/folding, general setup

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(123)

# Load data ----
load(here("data/train_clean.rda"))

# split into training/testing sets to assess and compare the ensemble model

# Data prep ----

# keep using the log transformation, and remove price this time
airbnb <- train_clean |> 
  mutate(price_log10 = log10(price)) |> 
  select(-price)

# skimr::skim_without_charts(airbnb)

# Split data ----

# use a 80/20 ratio, stratify by target variable
airbnb_split <- airbnb |> 
  initial_split(prop = 0.8, strata = price_log10)

# get split datasets
airbnb_train <- airbnb_split |> training()
airbnb_test <- airbnb_split |> testing()

# Fold data ----

# use 8 folds, 4 repeats, stratify by target variable
airbnb_folds <- airbnb_train |> 
  vfold_cv(
    v = 8,
    repeats = 4,
    strata = price_log10
  )

# Set up metrics ----
my_metrics <- metric_set(mae, rmse, rsq)

# Save objects ----
save(airbnb, file = here("attempt_3/data_splits/airbnb.rda"))
save(airbnb_train, file = here("attempt_3/data_splits/airbnb_train.rda"))
save(airbnb_test, file = here("attempt_3/data_splits/airbnb_test.rda"))
save(airbnb_folds, file = here("attempt_3/data_splits/airbnb_folds.rda"))
save(my_metrics, file = here('attempt_3/data_splits/my_metrics.rda'))
