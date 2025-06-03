# Classification Prediction Problem ----
# Stat 301-3
# Final model 2
# Step 1: data splitting/folding, general setup

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(34821)

# Load data ----
load(here("data/train_clean.rda"))

# split into training/testing sets to assess and compare the ensemble model

# Data prep ----

# add the host_match column
airbnb <- train_clean |> 
  add_count(host_about, host_since) |> 
  rename(host_match = n)

# skimr::skim_without_charts(airbnb)

# Split data ----

# use a 80/20 ratio, stratify by target variable
airbnb_split <- airbnb |> 
  initial_split(prop = 0.8, strata = host_is_superhost)

# get split datasets
airbnb_train <- airbnb_split |> training()
airbnb_test <- airbnb_split |> testing()

# Fold data ----

# use 8 folds, 4 repeats, stratify by target variable
airbnb_folds <- airbnb_train |> 
  vfold_cv(
    v = 8,
    repeats = 4,
    strata = host_is_superhost
  )

# Save objects ----
save(airbnb, file = here("final_submissions/model_2/data_splits/airbnb.rda"))
save(airbnb_train, file = here("final_submissions/model_2/data_splits/airbnb_train.rda"))
save(airbnb_test, file = here("final_submissions/model_2/data_splits/airbnb_test.rda"))
save(airbnb_folds, file = here("final_submissions/model_2/data_splits/airbnb_folds.rda"))
