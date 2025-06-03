# Classification Prediction Problem ----
# Stat 301-3
# Final model 2
# Step 2: setup preprocessing recipes

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle conflicts
tidymodels_prefer()

# Load data ----
load(here("final_submissions/model_2/data_splits/airbnb_train.rda"))

# Recipe base ----
base_rec <- recipe(host_is_superhost ~ ., data = airbnb_train) |> 
  step_mutate(
    location_match = case_when(
      stringr::str_detect(host_location, ", IL$|Illinois") & listing_location == "chicago" ~ 1,
      stringr::str_detect(host_location, ", HI$|Hawaii") & listing_location == "kauai" ~ 1,
      stringr::str_detect(host_location, ", NC$|North Carolina") & listing_location == "asheville" ~ 1,
      .default = 0
    ),
    exclam_desc = stringr::str_detect(description, "!"),
    host_info = !is.na(host_about),
    entire = stringr::str_detect(property_type, "[Ee]ntire"),
    num_amenities = stringr::str_count(amenities, ","),
    pool = stringr::str_detect(stringr::str_to_lower(amenities), "pool"),
    extra = stringr::str_detect(stringr::str_to_lower(amenities), "extra"),
    dishwasher = stringr::str_detect(stringr::str_to_lower(amenities), "dishwasher"),
    dedicated = stringr::str_detect(stringr::str_to_lower(amenities), "dedicated"),
    private = stringr::str_detect(stringr::str_to_lower(amenities), "private"),
    storage = stringr::str_detect(stringr::str_to_lower(amenities), "storage"),
    outdoor = stringr::str_detect(stringr::str_to_lower(amenities), "outdoor"),
    shades = stringr::str_detect(stringr::str_to_lower(amenities), "shades"),
    beach = stringr::str_detect(stringr::str_to_lower(amenities), "beach"),
    grill = stringr::str_detect(stringr::str_to_lower(amenities), "grill"),
    pets = stringr::str_detect(stringr::str_to_lower(amenities), "pets"),
    host_year = lubridate::year(host_since),
    longevity = lubridate::time_length(last_review - first_review, unit = "years"),
    across(where(is.numeric), \(x) if_else(x > 1000000, NA, x)),
    across(where(is.logical), as.numeric),
    bed_room_ratio = if_else(
      bedrooms == 0, beds, beds / bedrooms
    ),
    bed_bath_ratio = if_else(
      bathrooms == 0, bedrooms, bedrooms / bathrooms
    ),
    nights_range = maximum_nights - minimum_nights,
  ) |> 
  step_rm(
    where(lubridate::is.Date), description, host_location, host_about, host_neighbourhood,
    neighbourhood_cleansed, property_type, bathrooms_text, amenities
  ) |> 
  step_discretize(longevity, reviews_per_month, starts_with("review_scores_"), num_breaks = 3) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_other(host_response_time) |> 
  step_impute_mode(all_nominal_predictors())

# Tree recipe ----

# only including recipe for bt models
tree_rec <- base_rec |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_predictors())

# # check recipe
# tree_rec |>
#   prep() |>
#   bake(new_data = NULL) |>
#   skimr::skim_without_charts()

# 99 predictor columns after preprocessing

# save recipe
save(tree_rec, file = here("final_submissions/model_2/recipes/tree_rec.rda"))
