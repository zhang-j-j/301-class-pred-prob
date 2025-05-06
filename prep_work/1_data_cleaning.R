# Regression Prediction Problem ----
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
# skimr::skim_without_charts(train_raw)

# 57 columns, 9410 rows
# some missingness issues throughout

# retype variables:
# price (character to numeric)
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
    price = parse_number(price),
    bathrooms = if_else(
      # this gives a warning, but not an actual issue
      str_starts(bathrooms_text, "\\d"), parse_number(bathrooms_text),
      0.5
    ),
    across(contains("_rate"), parse_number),
    across(where(is.logical), as.factor),
    across(c(listing_location, room_type), as.factor),
    has_availability = fct_na_value_to_level(has_availability, level = "FALSE")
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
#   gg_miss_upset(nsets = 13)

# Notes ----

## Missingness ----

# missingness in host info: something to worry about later, since these are not
# really usable as predictors right now (need further processing)

# missingness in reviews: might be worthwhile to add a new variable indicating if
# reviews are recorded

# other variables have minor missingness that can simply be imputed/handled in
# preprocessing steps

## Variables ----

# for the very basic models: remove all character, date variables
# might contain information worth recovering later

# can probably get away with using everything else right now, need to impute

# Target variable check ----

# train_clean |> 
#   ggplot(aes(price)) +
#   geom_density()
# 
# train_clean |> 
#   ggplot(aes(price)) +
#   geom_boxplot()
# 
# # there is a strong right-skew, so will need to use some transformation
# # probably start off with log-transformation, maybe try more complex later
# 
# # log-transformation gives a pretty decent distribution
# train_clean |> 
#   ggplot(aes(price)) +
#   geom_density() +
#   scale_x_log10()

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
    across(c(listing_location, room_type), as.factor),
    has_availability = fct_na_value_to_level(has_availability, level = "FALSE")
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

# # examine semi-clean data: no major issues
# skimr::skim_without_charts(test_clean)

# Write out results ----
save(train_clean, file = here("data/train_clean.rda"))
save(test_clean, file = here("data/test_clean.rda"))
