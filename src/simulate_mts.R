rm(list=ls())
library("gratis")
library("forecast")
library("nortsTest")

# generate time series
x <- generate_ts_with_target(n = 2,
                             ts.length = 200,
                             freq = 1,
                             seasonal = 0,
                             features = c("tsfeatures", "heterogeneity"),
                             selected.features = c("trend", "linearity", "x_acf1", "arch_acf"),
                             target = c(0, 0, 0.5, 0.5),
                             parallel = FALSE)

# plot
# checking for ARMA components 
# 1. acf + pacf of the time series
ggtsdisplay(x[,1])
ggtsdisplay(x[,2])

# checking for heteroskedasticity
# 1. acf + pacf of the squares of the time series
ggtsdisplay(x[,1]^2)
ggtsdisplay(x[,2]^2)

# 2. lagrange multiplier test
Lm.test(y = x[,1]^2, lag.max = 4)
Lm.test(y = x[,2]^2, lag.max = 4)

y <- x[, 1] + x[, 2]

autoplot(y)

cor(y, x[, 2])
