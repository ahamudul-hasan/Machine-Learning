# Polynomial Regression

Polynomial regression is a way to fit a curved line to data.

It is still a type of linear regression, but instead of using only `x`, we also use powers of `x` like `x^2`, `x^3`, and so on.

## Simple idea

A straight line looks like this:

`y = mx + b`

A polynomial line can look like this:

`y = a*x^2 + b*x + c`

If the data is curved, a polynomial model can fit it much better than a straight line.

## Why use it?

Use polynomial regression when the relationship between input and output is not straight.

For example:
- house price growth that bends upward
- sales that rise and then slow down
- any pattern with a curve

## How it works

1. Start with the original feature `x`.
2. Create new features such as `x^2`, `x^3`, and more.
3. Train a linear regression model on those new features.
4. The model can now learn a curved pattern.

## Example from this notebook

This notebook uses a quadratic pattern:

`y = 0.8x^2 + 0.9x + 2 + noise`

That is why a degree-2 polynomial fit works well.

## Important note

- Low degree: may underfit the data.
- High degree: may overfit the data.

Usually, start with a small degree like `2` or `3` and increase only if needed.

## In simple words

Polynomial regression helps a model draw a curve instead of only a straight line.
It is useful when the data clearly bends.
