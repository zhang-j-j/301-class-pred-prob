# Classification Prediction Problem ----
# Stat 301-3
# Attempt 3
# Custom recipe step: step_host_match

# Load packages ----
library(dplyr)
library(recipes)

# Define custom recipe step ----

## this code was developed using AI assistance

# Constructor
step_host_match <- function(recipe, ..., role = "predictor", trained = FALSE,
                            columns = NULL, skip = FALSE, id = rand_id("host_match")) {
  terms <- enquos(...)
  add_step(
    recipe,
    step_host_match_new(terms = terms, role = role, trained = trained,
                        columns = columns, skip = skip, id = id)
  )
}

# Step object
step_host_match_new <- function(terms, role, trained, columns, skip, id) {
  step(
    subclass = "host_match",
    terms = terms,
    role = role,
    trained = trained,
    columns = columns,
    skip = skip,
    id = id
  )
}

# Prep method
prep.step_host_match <- function(x, training, info = NULL, ...) {
  col_names <- recipes_eval_select(x$terms, training, info)
  if (length(col_names) != 2) {
    rlang::abort("step_host_match requires exactly two columns.")
  }
  step_host_match_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    columns = names(col_names),
    skip = x$skip,
    id = x$id
  )
}

# Bake method
bake.step_host_match <- function(object, new_data, ...) {
  if (length(object$columns) != 2) {
    rlang::abort("You must select exactly two columns for host_match.")
  }
  
  col1 <- object$columns[1]
  col2 <- object$columns[2]
  
  counts <- new_data |>
    count(.data[[col1]], .data[[col2]], name = "host_match")
  
  new_data <- new_data |>
    left_join(counts, by = c(col1, col2))
  
  new_data
}
