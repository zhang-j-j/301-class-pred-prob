# Regression Prediction Problem ----
# Stat 301-3
# Attempt 1
# Step 5: fit selected models for submissions

# Load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(205)

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
cores <- parallel::detectCores(logical = FALSE) - 1
c1 <- makePSOCKcluster(cores)
registerDoParallel(c1)

## ols ----
ols_fit <- final_ols |> 
  fit(airbnb_train)

# write out results
save(ols_fit, file = here("attempt_1/submissions/fitted/ols_fit.rda"))

## en ----
en_fit <- final_en |> 
  fit(airbnb_train)

# write out results
save(en_fit, file = here("attempt_1/submissions/fitted/en_fit.rda"))

## linear knn ----
knn_lm_fits <- final_knn_lm |> 
  mutate(
    fit = map(final_knn_lm$workflow, \(x) fit(x, airbnb_train)) 
  )

# write out results
save(knn_lm_fits, file = here("attempt_1/submissions/fitted/knn_lm_fits.rda"))

## tree knn ----
knn_tree_fits <- final_knn_tree |> 
  mutate(
    fit = map(final_knn_tree$workflow, \(x) fit(x, airbnb_train)) 
  )

# write out results
save(knn_tree_fits, file = here("attempt_1/submissions/fitted/knn_tree_fits.rda"))

## rf ----

## bt ----
bt_fits <- final_bt |> 
  mutate(
    fit = map(final_bt$workflow, \(x) fit(x, airbnb_train)) 
  )

# write out results
save(bt_fits, file = here("attempt_1/submissions/fitted/bt_fits.rda"))

## poly svm ----

## rbf svm ----
svm_rbf_fit <- final_svm_rbf |> 
  fit(airbnb_train)

# write out results
save(svm_rbf_fit, file = here("attempt_1/submissions/fitted/svm_rbf_fit.rda"))

## mars ----
mars_fits <- final_mars |> 
  mutate(
    fit = map(final_mars$workflow, \(x) fit(x, airbnb_train)) 
  )

# write out results
save(mars_fits, file = here("attempt_1/submissions/fitted/mars_fits.rda"))

## nn ----
nn_fit <- final_nn |> 
  fit(airbnb_train)

# write out results
save(nn_fit, file = here("attempt_1/submissions/fitted/nn_fit.rda"))

# reset to sequential processing
stopCluster(c1)
registerDoSEQ()
rm(c1, cores)
