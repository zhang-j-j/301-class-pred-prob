# Classification Prediction Problem ----
# Stat 301-3
# Prep work
# Step 2: general exploratory data analysis
# same EDA appears to work well from the regression problem

# Load packages ----
library(tidyverse)
library(here)
library(tidymodels)
library(textrecipes)

# handle conflicts
tidymodels_prefer()

# Load data ----
load(here("data/train_clean.rda"))

# initial check
skimr::skim_without_charts(train_clean)

# Additional variables ----

## string columns ----

# variable for mismatch between listing and host location
train_clean |> 
  mutate(
    location_match = case_when(
      str_detect(host_location, ", IL$|Illinois") & listing_location == "chicago" ~ 1,
      str_detect(host_location, ", HI$|Hawaii") & listing_location == "kauai" ~ 1,
      str_detect(host_location, ", NC$|North Carolina") & listing_location == "asheville" ~ 1,
      .default = 0
    ),
    .keep = "used"
  ) |> 
  count(location_match)

# exclamation point in description (need to handle NA)
train_clean |> 
  mutate(
    exclam_desc = str_detect(description, "!"),
    .keep = "used"
  ) |> 
  count(exclam_desc)

# convert host_about to levels of has/does not have
train_clean |> 
  mutate(
    host_info = !is.na(host_about),
    .keep = "used"
  ) |> 
  count(host_info)

# neighborhood does not look recoverable
train_clean |> 
  count(neighbourhood_cleansed) |> 
  arrange(-n)

# consider if the property type is entire/not
train_clean |> 
  mutate(
    entire = str_detect(property_type, "[Ee]ntire"),
    .keep = "used"
  ) |> 
  count(entire)

## amenities ----

# count number of amenities (plus 1, not exact but should be fine)
train_clean |> 
  mutate(
    num_amenities = str_count(amenities, ","),
    .keep = "used"
  ) |> 
  count(num_amenities) |> 
  ggplot(aes(num_amenities, n)) +
  geom_line()

# # try tokenizing amenities (probably beter to just do it manually)
# recipe(price ~ ., data = train_clean) |> 
#   step_tokenize(amenities, custom_token = \(x) str_split(x, ", ")) |> 
#   show_tokens(amenities)

# detect specific amenities that split the dataset somewhat evenly
# this was done manually, might be missing some but usable for now
train_clean |> 
  mutate(
    pool = str_detect(str_to_lower(amenities), "pool"),
    extra = str_detect(str_to_lower(amenities), "extra"),
    dishwasher = str_detect(str_to_lower(amenities), "dishwasher"),
    dedicated = str_detect(str_to_lower(amenities), "dedicated"),
    private = str_detect(str_to_lower(amenities), "private"),
    storage = str_detect(str_to_lower(amenities), "storage"),
    outdoor = str_detect(str_to_lower(amenities), "outdoor"),
    shades = str_detect(str_to_lower(amenities), "shades"),
    beach = str_detect(str_to_lower(amenities), "beach"),
    grill = str_detect(str_to_lower(amenities), "grill"),
    pets = str_detect(str_to_lower(amenities), "pets"),
    .keep = "used"
  ) |> 
  mutate(across(where(is.logical), as.numeric))

## date columns ----

# host year
train_clean |> 
  mutate(
    year = year(host_since),
    .keep = "used"
  )

# years between first and last reviews (discretize this to account for no reviews)
train_clean |> 
  mutate(
    longevity = time_length(last_review - first_review, unit = "years"),
    .keep = "used"
  ) |> 
  ggplot(aes(longevity)) +
  geom_density()

## numeric columns ----

# fix outliers in minimum/maximum_nights variables (coerce to NA)
train_clean |> 
  mutate(across(where(is.numeric), \(x) if_else(x > 1000000, NA, x))) |> 
  skimr::skim_without_charts()
