# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 2: setup preprocessing recipes

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# Load data ----
load(here("attempt_1/data_splits/airbnb_train.rda"))

# Linear recipe ----

# recipe for linear, knn, svm models
lm_rec <- recipe(price_log10 ~ ., data = airbnb_train) |> 
  step_rm(
    where(lubridate::is.Date), description, host_location, host_about, 
    host_neighbourhood, neighbourhood_cleansed, property_type, bathrooms_text,
    amenities, starts_with("review_scores"), reviews_per_month
  ) |> 
  step_rm(price, skip = TRUE) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_unknown(host_response_time) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_predictors())

# # check recipe
# lm_rec |>
#   prep() |>
#   bake(new_data = NULL) |>
#   skimr::skim_without_charts()

# 39 predictor columns after preprocessing

# save recipe
save(lm_rec, file = here("attempt_1/recipes/lm_rec.rda"))

# Tree recipe ----

# recipe for knn, random forest, boosted tree, neural network models
tree_rec <- recipe(price_log10 ~ ., data = airbnb_train) |> 
  step_rm(
    where(lubridate::is.Date), description, host_location, host_about, host_neighbourhood,
    neighbourhood_cleansed, property_type, bathrooms_text, amenities,
    starts_with("review_scores"), reviews_per_month
  ) |> 
  step_rm(price, skip = TRUE) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_unknown(host_response_time) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_predictors())

# # check recipe
# tree_rec |>
#   prep() |>
#   bake(new_data = NULL) |>
#   skimr::skim_without_charts()

# 43 predictor columns after preprocessing

# save recipe
save(tree_rec, file = here("attempt_1/recipes/tree_rec.rda"))
