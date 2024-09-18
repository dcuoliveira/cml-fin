# Copyright (c) 2017
# All rights reserved.  See the file COPYING for license terms. 


####################################
### Monetary Policy Example
###################################

rm(list=ls())

## load libraries
library(MASS)
library(vars)
library(parallel)
library(seqICP)

library(here)
library(dplyr)
library(data.table)
library(lubridate)

## load MPdata
load(here("src", "new_studies", "processedData", "monetarypolicy.Rda"))

###
# Extract variables
###

date <- MPdata[,1]
EXMA <- MPdata[,2]
EXME <- MPdata[,3]
EXMAUS <- MPdata[,4]
EXMEUS <- MPdata[,5]
FCI <- MPdata[,6]
Gold <- MPdata[,7]
RIMF <- MPdata[,8]
CHFsec <- MPdata[,9]
MA <- MPdata[,10]
OA <- MPdata[,11]
Assets <- MPdata[,12]
CPI <- MPdata[,13]
BCI <- MPdata[,14]
CMR <- MPdata[,15]
EuroGDP <- MPdata[,16]
SwissGDP <- MPdata[,17]

## 
# Transform variables
##

totOA <- Assets-MA-RIMF-CHFsec
ldtotOAr <- diff(log(totOA/Assets))

# log returns
ldEXME <- diff(log(1/EXME))
# log differences of GDP with currency removed for Swiss GDP
ldGDPeu <- diff(log(EuroGDP))
ldGDPch <- diff(log(SwissGDP/EXMA))
# log-differences of fraction of balance sheet
ldFCIr <- diff(log(FCI/Assets))
ldRIMFr <- diff(log(RIMF/Assets))
ldCHFr <- diff(log(CHFsec/Assets))
ldGoldr <- diff(log(Gold/Assets))
ldOAr <- diff(log(OA/Assets))
ldMAr <- diff(log(MA/Assets))
# compute inflation from CPI
dCPI <- diff(CPI)/(CPI[-1])
# difference of call money rate (log not possible due to negative rates)
dCMR <- diff(CMR)

##
# Collect relevant variables
##

MPdata2 <- cbind(ldEXME,dCMR,ldFCIr,ldRIMFr,ldMAr,ldCHFr,ldtotOAr,ldGDPch,ldGDPeu,dCPI)
MPdata2 <- scale(MPdata2)
MPdata3 <- MPdata2 %>% as.data.table() %>% mutate(date=my(date[2:length(date)])) %>% select(date, everything())
y.ind <- 1
X <- MPdata2[,-y.ind]
Y <- as.numeric(MPdata2[,y.ind])

write.csv(MPdata3, here("src", "new_studies", "processedData", "monetary-policy-processed.csv"))

## ##
## # Time series plot
## ##

## pdf("timeseries_plot.pdf",pointsize=6)
## xa <- date[-1]
## len <- length(xa)
## at <- seq(12,len,by=12)
## par(mfrow=c(5,2),mar=c(4.5,6,2,1)+0.1)
## plot(xa,Y,type="l",xlab="year",xaxt="n",cex.lab = 1.8)
## axis(side = 1, at = xa[at])
## for(i in 1:9){
##   plot(xa,X[,i],type="l",xlab="year",ylab=paste("X",toString(i),sep=""),xaxt="n",cex.lab = 1.8)
##   axis(side = 1, at = xa[at])
## }
## dev.off()


##
# Apply icps using different lags (using smooth.variance)
##

lags <- 1:6
res <- vector("list",length(lags))
pvals <- matrix(NA,length(lags),ncol(MPdata2)-1)

mcfun <- function(lags){
  res <- seqICP(X,Y,test="smooth.variance",par.test=list(alpha=0.05,B=1000),model="ar",par.model=list(pknown=TRUE,p=lags),stopIfEmpty=FALSE,silent=TRUE)
  print("one step")
  return(res)
}

res <- mclapply(lags,mcfun,mc.cores=2)
save(res,file="smooth.variance_lags.Rda")
for(i in 1:length(lags)){
  pvals[i,] <- res[[i]]$p.values
}
matplot(pvals)

##
# Apply icps using different lags (using two grid points and variance)
##

lags2 <- 1:6
pvals2 <- matrix(NA,length(lags2),ncol(MPdata2)-1)

mcfun <- function(lags){
  res <- seqICP(X,Y,test="variance",par.test=list(alpha=0.05,B=1000,link=sum, grid=c(0,70,140,nrow(MPdata2)),complements=TRUE),model="ar",par.model=list(pknown=TRUE,p=lags),stopIfEmpty=FALSE,silent=TRUE)
  print("one step")
  return(res)
}

res2 <- mclapply(lags2,mcfun,mc.cores=2)
save(res2,file="variance_lags.Rda")
for(i in 1:length(lags2)){
  pvals2[i,] <- res2[[i]]$p.values
}
matplot(pvals2)


##
# Apply icps using different lags (using two grid points and decoupled)
##

lags3 <- 1:6
pvals3 <- matrix(NA,length(lags3),ncol(MPdata2)-1)

mcfun <- function(lags){
  res <- seqICP(X,Y,test="decoupled",par.test=list(alpha=0.05,B=1000,link=sum, grid=c(0,70,140,nrow(MPdata2)),complements=TRUE),model="ar",par.model=list(pknown=TRUE,p=lags),stopIfEmpty=FALSE,silent=TRUE)
  print("one step")
  return(res)
}

res3 <- mclapply(lags3,mcfun,mc.cores=2)
save(res3,file="decoupled_lags.Rda")
for(i in 1:length(lags3)){
  pvals3[i,] <- res3[[i]]$p.values
}
matplot(pvals3)


##
# Apply icps using different lags (using two grid points and combined)
##

lags4 <- 1:6
pvals4 <- matrix(NA,length(lags4),ncol(MPdata2)-1)

mcfun <- function(lags){
  res <- seqICP(X,Y,test="combined",par.test=list(alpha=0.05,B=1000,link=sum, grid=c(0,70,140,nrow(MPdata2)),complements=TRUE),model="ar",par.model=list(pknown=TRUE,p=lags),stopIfEmpty=FALSE,silent=TRUE)
  print("one step")
  return(res)
}

res4 <- mclapply(lags4,mcfun,mc.cores=2)
save(res4,file="combined_lags.Rda")
for(i in 1:length(lags4)){
  pvals4[i,] <- res4[[i]]$p.values
}
matplot(pvals4)



##
# Apply icps using different lags (using hsic)
##

lags5 <- 1:10
pvals5 <- matrix(NA,length(lags5),ncol(MPdata2)-1)

mcfun <- function(lags){
  res <- seqICP(X,Y,test="hsic",par.test=list(alpha=0.05,B=1000),model="ar",par.model=list(pknown=TRUE,p=lags),stopIfEmpty=FALSE,silent=TRUE)
  print("one step")
  return(res)
}

res5 <- mclapply(lags5,mcfun,mc.cores=2)
save(res5,file="hsic_lags.Rda")
for(i in 1:length(lags5)){
  pvals5[i,] <- res5[[i]]$p.values
}
matplot(pvals5)




##
# Apply icps with different test statistics
##

## "variance" test statistic

res1 <- seqICP(X,Y,test="variance",par.test=list(grid=c(0,70,140,nrow(MPdata2)),complements=TRUE, alpha=0.05,B=1000,link=sum),model="ar",par.model=list(pknown=TRUE,p=5),stopIfEmpty=FALSE,silent=FALSE)
summary(res1)

save(res1,file="variance5.Rda")

## "decoupled" test statistic
res2 <- seqICP(X,Y,test="decoupled",par.test=list(grid=c(1,70,140,nrow(MPdata2)),complements=TRUE,alpha=0.05,B=1000,link=sum),model="ar",par.model=list(pknown=TRUE,p=5),stopIfEmpty=FALSE,silent=FALSE)
summary(res2)

save(res2,file="decoupled5.Rda")

## "combined" test statistic
res3 <- seqICP(X,Y,test="combined",par.test=list(grid=c(1,70,140,nrow(MPdata2)),complements=TRUE,alpha=0.05,B=1000,link=sum),model="ar",par.model=list(pknown=TRUE,p=5),stopIfEmpty=FALSE,silent=FALSE)
summary(res3)

save(res3,file="combined5.Rda")

## "smooth variance" test statistic
res4 <- seqICP(X,Y,test="smooth.variance",par.test=list(alpha=0.05,B=1000),model="ar",par.model=list(pknown=TRUE,p=5),stopIfEmpty=FALSE,silent=FALSE)
summary(res4)

save(res4,file="smooth.variance5.Rda")


## "hsic" test statistic
res5 <- seqICP(X,Y,test="hsic",par.test=list(alpha=0.05,B=1000),model="ar",par.model=list(pknown=TRUE,p=5),stopIfEmpty=FALSE,silent=FALSE)
summary(res5)

save(res5,file="hsic5.Rda")


##
# Fit a linear regression model with all instantaneous effects
##

n <- length(Y)
max.p <- 10
d <- ncol(X)
AICs <- numeric(max.p+1)
lin.fits <- vector("list",max.p+1)
pvals_lin <- matrix(NA,d,max.p+1)
# no lags
lin.fits[[1]] <- lm(Y~X)
k <- length(lin.fits[[1]]$coefficients)
nn <- nobs(lin.fits[[1]])
AICs[1] <- AIC(lin.fits[[1]])+2*k*(k+1)/(nn-k-1)
pvals_lin[,1] <- summary(lin.fits[[1]])$coefficients[-1,4]
# lagged models
for(i in 1:max.p){
  # generate design matrix
  Xdesign <- X[(i+1):n,]
  for(j in 1:i){
    Xdesign <- cbind(Xdesign,Y[(i+1-j):(n-j)],X[(i+1-j):(n-j),])
  }
  # compute linear model and AIC
  lin.fits[[i+1]] <- lm(Y[(i+1):n]~Xdesign)
  k <- length(lin.fits[[i+1]]$coefficients)
  nn <- nobs(lin.fits[[i+1]])
  AICs[i+1] <- AIC(lin.fits[[i+1]])+2*k*(k+1)/(nn-k-1)
  pvals_lin[,i+1] <- summary(lin.fits[[i+1]])$coefficients[2:(d+1),4]
}

matplot(t((pvals_lin)))
abline(h=0.05)

which.min(AICs)-1
summary(lin.fits[[which.min(AICs)]])
pvals_lin[,which.min(AICs)]

save(pvals_lin,file="linear_lags.Rda")
