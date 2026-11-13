# Reference values for tests/test_survey_reference.py, from R's survey package.
# Rerun with: Rscript tests/fixtures/make_survey_reference.R
library(survey)
library(jsonlite)

set.seed(20261112)
n_psu <- 24
psu <- rep(seq_len(n_psu), times = sample(3:9, n_psu, replace = TRUE))
n <- length(psu)
stratum <- c("a", "b", "c")[(psu - 1) %% 3 + 1]
cell <- sample(c("residential", "tertiary", "primary"), n, replace = TRUE, prob = c(.5, .3, .2))
psu_effect <- rnorm(n_psu)[psu]
z <- rpois(n, 3 + (cell == "residential") + 0.5 * (stratum == "a"))
y <- rbinom(n, z, plogis(-1.2 + 0.4 * psu_effect + 0.5 * (cell == "residential")))
w <- round(runif(n, 0.5, 4) * ifelse(stratum == "c", 2, 1), 3)
level <- round(60 + 6 * psu_effect + 4 * (cell == "primary") + rnorm(n, sd = 5), 1)
data <- data.frame(psu, stratum, cell, y, z, w, level)
write.csv(data, "tests/fixtures/survey_reference.csv", row.names = FALSE)

pair <- function(stat) list(estimate = unname(coef(stat))[1], se = unname(SE(stat))[1])
report <- function(design) list(
  mean = pair(svymean(~level, design)),
  ratio = pair(svyratio(~y, ~z, design)),
  total = pair(svytotal(~y, design)),
  share_at_least = pair(svymean(~I(as.numeric(level >= 62)), design))
)

shares <- c(residential = 0.6, tertiary = 0.25, primary = 0.15)
clustered <- svydesign(ids = ~psu, weights = ~w, data = data)
population <- data.frame(cell = names(shares), Freq = shares * sum(data$w))

# Shares fix the mix of cells, not the population size, so a post-stratified
# total is the estimated size times the post-stratified mean. postStratify()
# would treat the size as known; the delta method keeps its sampling error.
cell_totals <- data.frame(
  y_res = y * (cell == "residential"), n_res = as.numeric(cell == "residential"),
  y_ter = y * (cell == "tertiary"), n_ter = as.numeric(cell == "tertiary"),
  y_pri = y * (cell == "primary"), n_pri = as.numeric(cell == "primary")
)
by_cell <- svytotal(
  ~y_res + n_res + y_ter + n_ter + y_pri + n_pri,
  svydesign(ids = ~psu, weights = ~w, data = cbind(data, cell_totals))
)
total_size_estimated <- pair(svycontrast(by_cell, quote(
  (n_res + n_ter + n_pri) *
    (0.6 * y_res / n_res + 0.25 * y_ter / n_ter + 0.15 * y_pri / n_pri)
)))

reference <- list(
  shares = as.list(shares),
  weights = report(svydesign(ids = ~1, weights = ~w, data = data)),
  weights_cluster = report(clustered),
  weights_strata = report(svydesign(ids = ~1, strata = ~stratum, weights = ~w, data = data)),
  weights_strata_cluster = report(svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = data)),
  post_stratified = report(postStratify(clustered, ~cell, population))
)
reference$post_stratified$total <- total_size_estimated
write_json(reference, "tests/fixtures/survey_reference.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
