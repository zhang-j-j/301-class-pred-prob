# Classification Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 5: fit selected models for submissions

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(future)
library(discrim)

# handle conflicts
tidymodels_prefer()

# Load objects ----

# training data
load(here("attempt_1/data_splits/airbnb_train.rda"))

# final workflows
list.files(
  path = here("attempt_1/submissions/workflows/"),
  pattern = ".rda",
  full.names = TRUE
) |> 
  walk(\(x) load(x, envir = globalenv()))

# Fit final workflows ----

# set up parallel processing
cores <- availableCores() - 1
plan(multisession, workers = cores)

# comment out fits to avoid refitting/changing models

## nb ----
# nb_fit <- final_nb |> 
#   fit(airbnb_train)
# 
# # write out results
# save(nb_fit, file = here("attempt_1/submissions/fitted/nb_fit.rda"))

## log ----
# log_fit <- final_log |> 
#   fit(airbnb_train)
# 
# # write out results
# save(log_fit, file = here("attempt_1/submissions/fitted/log_fit.rda"))

## en ----
# en_fit <- final_en |> 
#   fit(airbnb_train)
# 
# # write out results
# save(en_fit, file = here("attempt_1/submissions/fitted/en_fit.rda"))

## linear knn ----
# set seed (since these may be run separately)
set.seed(28)

knn_lm_fits <- final_knn_lm |>
  mutate(
    fit = map(final_knn_lm$workflow, \(x) fit(x, airbnb_train))
  )

# write out results
save(knn_lm_fits, file = here("attempt_1/submissions/fitted/knn_lm_fits.rda"))

## tree knn ----
# set seed (since these may be run separately)
set.seed(81)

knn_tree_fits <- final_knn_tree |>
  mutate(
    fit = map(final_knn_tree$workflow, \(x) fit(x, airbnb_train))
  )

# write out results
save(knn_tree_fits, file = here("attempt_1/submissions/fitted/knn_tree_fits.rda"))

## rf ----
# set seed (since these may be run separately)
set.seed(891)

rf_fits <- final_rf |>
  mutate(
    fit = map(final_rf$workflow, \(x) fit(x, airbnb_train))
  )

# write out results
save(rf_fits, file = here("attempt_1/submissions/fitted/rf_fits.rda"))

## bt ----
# # set seed (since these may be run separately)
# set.seed(25)
# 
# bt_fits <- final_bt |> 
#   mutate(
#     fit = map(final_bt$workflow, \(x) fit(x, airbnb_train)) 
#   )
# 
# # write out results
# save(bt_fits, file = here("attempt_1/submissions/fitted/bt_fits.rda"))

# rbf svm ----
svm_rbf_fit <- final_svm_rbf |>
  fit(airbnb_train)

# write out results
save(svm_rbf_fit, file = here("attempt_1/submissions/fitted/svm_rbf_fit.rda"))

## mars ----
# mars_fits <- final_mars |> 
#   mutate(
#     fit = map(final_mars$workflow, \(x) fit(x, airbnb_train)) 
#   )
# 
# # write out results
# save(mars_fits, file = here("attempt_1/submissions/fitted/mars_fits.rda"))

## nn ----
# nn_fit <- final_nn |> 
#   fit(airbnb_train)
# 
# # write out results
# save(nn_fit, file = here("attempt_1/submissions/fitted/nn_fit.rda"))

# reset to sequential processing
plan(sequential)
