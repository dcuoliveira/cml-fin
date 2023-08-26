# ---- [PART 1] Variance Shifted VAR(1) ----

set.seed(1)
# na <- 140
# nb <- 80
na <- 1400
nb <- 800
n <- na + nb

x1 <- matrix(data = NA, nrow = n, ncol = 1)
x2 <- matrix(data = NA, nrow = n, ncol = 1)
x3 <- matrix(data = NA, nrow = n, ncol = 1)

for (i in 1:n){
  
  # initialize all series
  if (i == 1){
    x1[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    next
  }
  
  # sample time series iteratively conditioned on each environment
  if (i <= na){
    x1[i,] <- 0.2*x1[i-1,] + 0.7*x3[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- 0.7*x1[i-1,] + 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
  }
  else{
    x1[i,] <- 0.2*x1[i-1,] + 0.7*x3[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- 0.7*x1[i-1,] + 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 3)
  }
  
}

var_sim_shifted <- cbind(x1, x2, x3)
ts.plot(var_sim_shifted, type="l", col=c(1, 2, 3))

var_sim_shifted_df <- var_sim_shifted %>% as.data.table()
colnames(var_sim_shifted_df) <- c("x1", "x2", "x3")
varnames <- colnames(var_sim_shifted_df)

# ---- seqICP ----

library(seqICP)
library(dplyr)
library(data.table)

# SH: let's check each target one by one. For sanity checks, let's assume 
#.  p = 1 is known.

# (1) SH: Y (x1) has a parent x3 of lag 1 and a parent of itself of lag 1.
vn <- "x1"
X <- var_sim_shifted_df %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df %>% dplyr::select(one_of(vn))
output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output) 
# SH: nothing is found, but this is as expected, as there is no instantaneous 
#.  effect from the covariates. Note: S = {} corresponds to fitting a model 
#.  with only the lag 1 of Y and X; S = {1} corresponds to fitting a model with 
#.  the current X1 and the lag 1 of Y and X, etc. 

## (2) SH: Y (x2) has only one parent, itself of lag 1.
vn <- "x2"
X <- var_sim_shifted_df %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df %>% dplyr::select(one_of(vn))
output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)
# SH: nothing is found, as there is again no instantaneous effect. 

## (3) SH: Y (x3) has one parent x1 with lag 1 and one parent x2 with lag 1.
vn <- "x3"
X <- var_sim_shifted_df %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df %>% dplyr::select(one_of(vn))
output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)
# SH: since the target x3 is intervened directly, as expected, no invariant set 
#.  can be found. 

# ---- Variance Shifted VAR(1) (not intervene on x3) ----

set.seed(1)
# na <- 140
# nb <- 80
na <- 1400
nb <- 800
n <- na + nb

x1 <- matrix(data = NA, nrow = n, ncol = 1)
x2 <- matrix(data = NA, nrow = n, ncol = 1)
x3 <- matrix(data = NA, nrow = n, ncol = 1)

for (i in 1:n){
  
  # initialize all series
  if (i == 1){
    x1[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    next
  }
  
  # sample time series iteratively conditioned on each environment
  if (i <= na){
    x1[i,] <- 0.2*x1[i-1,] + 0.7*x3[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- 0.7*x1[i-1,] + 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
  } 
  else {
    x1[i,] <- 0.2*x1[i-1,] + 0.7*x3[i-1,] + rnorm(n = 1, mean = 0, sd = 5)
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 5)
    x3[i,] <- 0.7*x1[i-1,] + 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
  }
  
}

var_sim_shifted <- cbind(x1, x2, x3)
ts.plot(var_sim_shifted, type="l", col=c(1, 2, 3))

var_sim_shifted_df <- var_sim_shifted %>% as.data.table()
colnames(var_sim_shifted_df) <- c("x1", "x2", "x3")
varnames <- colnames(var_sim_shifted_df)

## SH: as before, Y (x3) has one parent x1 with lag 1 and one parent x2 with lag 1.
vn <- "x3"
X <- var_sim_shifted_df %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df %>% dplyr::select(one_of(vn))
output <- seqICP(X=X, Y=y, test = "variance", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)
# SH: there still shouldn't be any significant (instantaneous) causal effect, 
#.  when I tried with the original samle size, I think it's because of the 
#.  randomness in the test that sometimes it returns x1 and x2 -- 
#.  this has improved after I increased the sample size.

# SH: what if we run it with a manual regressor matrix with only the past?
X <- cbind(X[1:(nrow(X)-1),])
colnames(X) <- c("x1_1", "x2_1")
y <- y[2:nrow(y),]
output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar",
                 par.model = list(pknown = TRUE, p = 0),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)
# SH: this should give back lag 1 of x1 and lag 1 of x2.

# ---- [PART 2] Variance Shifted SVAR(1) (Model C of section 6 from Pfister et al. (2018)) ----

set.seed(1)
# na <- 140
# nb <- 80
na <- 1400
nb <- 800

x1 <- matrix(data = NA, nrow = n, ncol = 1)
x2 <- matrix(data = NA, nrow = n, ncol = 1)
x3 <- matrix(data = NA, nrow = n, ncol = 1)

for (i in 1:n){
  
  # initialize all series
  if (i == 1){
    x1[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    next
  }
  
  # sample time series iteratively conditioned on each environment
  if (i <= na){
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- 0.2*x2[i,] + rnorm(n = 1, mean = 0, sd = 1)
    x1[i,] <- 0.7*x3[i,] + rnorm(n = 1, mean = 0, sd = 1)
  }
  else{
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- 0.2*x2[i,] + rnorm(n = 1, mean = 0, sd = 3)
    x1[i,] <- 0.7*x3[i,] + rnorm(n = 1, mean = 0, sd = 1)
  }
  
}

var_sim_shifted2 <- cbind(x1, x2, x3)
ts.plot(var_sim_shifted2, type="l", col=c(1, 2, 3))

var_sim_shifted_df2 <- var_sim_shifted2 %>% as.data.table()
colnames(var_sim_shifted_df2) <- c("x1", "x2", "x3")
varnames <- colnames(var_sim_shifted_df2)

# SH: I did get empty set when the sample size was small, but after increasing 
#.  the sample size x2 is indeed significant as expected. 
vn <- "x1"
X <- var_sim_shifted_df2 %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df2 %>% dplyr::select(one_of(vn))

output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)

# SH: since x2 doesn't have an instantaneous parent, this should indeed not 
#.  give anything significant.
vn <- "x2"
X <- var_sim_shifted_df2 %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df2 %>% dplyr::select(one_of(vn))

output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)

# SH: for x3, since it's intervened, we won't find anything either. But see 
#.  below for when the intervention is on x1 and x2
vn <- "x3"
X <- var_sim_shifted_df2 %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df2 %>% dplyr::select(one_of(vn))

output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)

# ---- [PART 2] Variance Shifted SVAR(1) (intervene on x1 and x2) ----

set.seed(1)
# na <- 140
# nb <- 80
na <- 1400
nb <- 800
n <- na + nb

x1 <- matrix(data = NA, nrow = n, ncol = 1)
x2 <- matrix(data = NA, nrow = n, ncol = 1)
x3 <- matrix(data = NA, nrow = n, ncol = 1)

for (i in 1:n){
  
  # initialize all series
  if (i == 1){
    x1[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x2[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- rnorm(n = 1, mean = 0, sd = 1)
    next
  }
  
  # sample time series iteratively conditioned on each environment
  if (i <= na){
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 1)
    x3[i,] <- 0.2*x2[i,] + rnorm(n = 1, mean = 0, sd = 1)
    x1[i,] <- 0.7*x3[i,] + rnorm(n = 1, mean = 0, sd = 1)
  }
  else{
    x2[i,] <- 0.2*x2[i-1,] + rnorm(n = 1, mean = 0, sd = 3)
    x3[i,] <- 0.2*x2[i,] + rnorm(n = 1, mean = 0, sd = 1)
    x1[i,] <- 0.7*x3[i,] + rnorm(n = 1, mean = 0, sd = 3)
  }
  
}

var_sim_shifted2 <- cbind(x1, x2, x3)
ts.plot(var_sim_shifted2, type="l", col=c(1, 2, 3))

var_sim_shifted_df2 <- var_sim_shifted2 %>% as.data.table()
colnames(var_sim_shifted_df2) <- c("x1", "x2", "x3")
varnames <- colnames(var_sim_shifted_df2)

vn <- "x3"
X <- var_sim_shifted_df2 %>% dplyr::select(-one_of(vn))
y <- var_sim_shifted_df2 %>% dplyr::select(one_of(vn))

# SH: now x3 indeed has x2 as the instantaneous effect
output <- seqICP(X=X, Y=y, test = "decoupled", model = "ar", 
                 par.model = list(pknown = TRUE, p = 1),
                 stopIfEmpty = FALSE, silent = FALSE)
print(paste0("Target: ", vn, " Features: ", paste(colnames(X), collapse = " ")))
summary(output)

