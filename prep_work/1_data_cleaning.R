# Classification Prediction Problem ----
# Stat 301-3
# Prep work
# Step 1: data check and data cleaning

# Load packages ----
library(tidyverse)
library(here)
library(naniar)

# Load data ----
train_raw <- read_csv(here("data/train.csv"), na = c("", "NA", "N/A"))
test_raw <- read_csv(here("data/test.csv"), na = c("", "NA", "N/A"))

# Data check ----
skimr::skim_without_charts(train_raw)

# 57 columns, 10500 rows
# some missingness issues throughout

# retype variables:
# host_is_superhost (numeric to factor)
# host_*_rate (character to numeric)
# change logicals to factor

# more advanced changes:
# host_verifications, amenities (extract from list)
# bathrooms_text (extract the number)

# other notes to be aware of:
# host_neighbourhood vs neighbourhood_cleansed
# property type vs room type
# minimum/maximum nights seem to be redundant

# Clean data ----
train_clean <- train_raw |> 
  janitor::clean_names() |> 
  mutate(
    host_is_superhost = factor(host_is_superhost, levels = c(1, 0)),
    bathrooms = if_else(
      # this gives a warning, but not an actual issue
      str_starts(bathrooms_text, "\\d"), parse_number(bathrooms_text),
      0.5
    ),
    across(contains("_rate"), parse_number),
    across(where(is.logical), as.factor),
    across(c(listing_location, room_type, host_response_time), as.factor),
    has_availability = fct_na_value_to_level(has_availability, level = "FALSE"),
    host_response_time = fct_na_value_to_level(host_response_time, level = "unknown")
  )

## Handle listed entries ----

# host_verifications is easy to manage now, too many amenities levels
train_clean <- train_clean |> 
  mutate(
    n = 1,
    across(
      c(host_verifications, amenities), 
      \(x) str_replace_all(x, "\\[|]|\"|'", "")
    )
  ) |> 
  separate_longer_delim(host_verifications, ", ") |> 
  filter(host_verifications %in% c("email", "phone", "work_email")) |> 
  pivot_wider(
    names_from = host_verifications,
    values_from = n,
    values_fill = 0
  )

# # examine semi-clean data
# skimr::skim_without_charts(train_clean)
# 
# train_clean |>
#   miss_var_summary() |>
#   print(n = 25)
# 
# train_clean |>
#   gg_miss_upset(nsets = 14)

# Notes ----

## Missingness ----

# missingness in host info: something to worry about later, since these are not
# really usable as predictors right now (need further processing)

# missingness in reviews: might be worthwhile to add a new variable indicating if
# reviews are recorded

# somewhat high missingness in beds, but still not terrible, can be handled
# other variables have minor missingness that can simply be imputed/handled in
# preprocessing steps

## Variables ----

# for the very basic models: remove all character, date variables
# might contain information worth recovering later

# can probably get away with using everything else right now, need to impute

# Target variable check ----

# train_clean |> 
#   ggplot(aes(host_is_superhost)) +
#   geom_bar()

# very even distribution between classes, so unlikely to need up/downsampling

# Clean testing set ----
test_clean <- test_raw |> 
  janitor::clean_names() |> 
  mutate(
    bathrooms = if_else(
      # this gives a warning, but not an actual issue
      str_starts(bathrooms_text, "\\d"), parse_number(bathrooms_text),
      0.5
    ),
    across(contains("_rate"), parse_number),
    across(where(is.logical), as.factor),
    across(c(listing_location, room_type, host_response_time), as.factor),
    has_availability = fct_na_value_to_level(has_availability, level = "FALSE"),
    host_response_time = fct_na_value_to_level(host_response_time, level = "unknown")
  )

test_clean <- test_clean |> 
  mutate(
    n = 1,
    across(
      c(host_verifications, amenities), 
      \(x) str_replace_all(x, "\\[|]|\"|'", "")
    )
  ) |> 
  separate_longer_delim(host_verifications, ", ") |>
  mutate(
    host_verifications = if_else(
      host_verifications == "", "None", 
      host_verifications
    )
  ) |> 
  pivot_wider(
    names_from = host_verifications,
    values_from = n,
    values_fill = 0
  ) |> 
  select(-None)

# # examine semi-clean data: there is some new missingness in the factor variables
# skimr::skim_without_charts(test_clean)

# Write out results ----
save(train_clean, file = here("data/train_clean.rda"))
save(test_clean, file = here("data/test_clean.rda"))
