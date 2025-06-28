"""

As a starting point let's consider this brms formula

brm(formula = time ~ age + disease + (1 | category)

"""

blf(y ~ a(age + diease) + c(category))

