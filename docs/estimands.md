# Choosing an estimand

An estimand states which population quantity the estimate targets. Counts can
produce means, totals, or ratios. The input type does not determine the target.

## Mean

`sample.mean("n_people")` estimates the average value per population unit. A
mean of a binary indicator is a population proportion.

## Population total

`sample.total("n_people", population_size=N)` estimates `N` times the sample
mean. `N` must count the same row-level population units represented by the
sample. The method requires `N` because observed counts alone do not identify a
population total.

The current variance estimator uses a with-replacement approximation. It does
not apply a finite population correction, even when the sample is a large part
of the population.

## Ratio of totals

`sample.ratio("n_women", "n_people")` estimates:

```text
sum(n_women) / sum(n_people)
```

Rows with larger denominators contribute more to the result. This is the usual
aggregate proportion when each row records a category count and a total count.

## Mean of row-level ratios

`sample.mean_of_ratios("n_women", "n_people")` estimates:

```text
mean(n_women / n_people)
```

Every row contributes equally. The method requires a positive denominator in
every row. It does not silently remove rows because doing so changes the target
population. Filter the DataFrame first when the intended population excludes
zero-denominator rows.

The ratio of totals and the mean of ratios answer different questions. They are
equal only in special cases, such as a constant denominator.
