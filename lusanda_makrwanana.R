# Name : Lusanda Makrwanana
# Institution: African institute for Mathematical Sciences
# Title : Forecasting the Price of Ethereum with Fractional
#         Ornstein–Uhlenbeck Lévy process: A Machine Learning Approach
# Last updated on 02 November 2023

# ************************************************************************************
# LOAD LIBRARIES
#*************************************************************************************
library(fracdiff)
library(fBasics)
library(caret)
library(quantmod) 
library(timeSeries)
library(PerformanceAnalytics)
library(yuima)
library(dygraphs)
library(quantmod) 
library(coinmarketcapr)
library(RCurl)
library(MASS)
library(psych)
library(tidyverse)
library(zoo)
library(expm)
library(cubature)
library(mvtnorm)
library(Metrics)
library(dplyr)
library(randomForest)
library(fitdistrplus)
library(ggplot2)

set.seed(1234)
#****************************************************************************************
# HISTORICAL DATA OF ETHEREUM FROM YAHOO FINANCE
#****************************************************************************************
ETH <- getSymbols("ETH-USD", src = "yahoo", to='2023-10-18', auto.assign = FALSE)
prices <-`ETH`$`ETH-USD.Close` # Extracting Closing Prices

# The returns of the ETH time series
returns <- na.omit(timeSeries::returns(prices, percentage = TRUE, c("continuous", "discrete", "compound", "simple")[1], trim=TRUE))
n <- length(prices) # The length of the prices

Z <- na.omit(diff(log(prices)))[-1] # Logarithmic returns.
Z1 <- Z-mean(Z)
m <- length(Z)

Z.0 <- Z[m] # Last value of the Z
Z <- as.zoo(Z)
Z1 <- as.zoo(Z1)
Z1.0 <- Z1[length(Z1)]

zMin <- min(Z) # Minimum of Z
zMax <- max(Z) # Maximum of Z

# Time step or the simulation.
Dt <- 1/365
#*************************************************************
# Step 2: Diffusion Process with Geometric Brownian motion estimation
#*************************************************************
model1 <- setModel(drift="mu*x", diff="sigma*x")
gBm <- setYuima(model=model1, data=setData(cumsum(Z),delta=Dt))
gBm.fit <- qmle(gBm, start=list(mu=0,sigma=1),method="BFGS")
gBm.cf <- coef(gBm.fit)

#*************************************************************
# Step 2.1 Compound Poisson with Gaussian-distribution without intensity
#*************************************************************
model3 <- setPoisson( df="dnorm(z,mu,sigma)")
Norm <- setYuima(model=model3, data=setData(cumsum(Z),delta=Dt))
Norm.fit <- qmle(Norm,start=list(mu=1, sigma=1),
                 lower=list(mu=1e-7,sigma=0.01),method="L-BFGS-B")
Norm.cf <- coef(Norm.fit)
#*************************************************************
# Step 2,2 Jumps with NIG-Levy distribution
#*************************************************************
model2 <- setPoisson(df="dNIG(z,alpha,beta,delta1,mu)")
NIG <- setYuima(model=model2,
                data=setData(cumsum(Z),delta=Dt))
NIG.fit <- qmle(NIG,start=list(alpha=5, beta=1, delta1=0.03,mu=0.0008),
                lower=list(alpha=1,beta=0, delta1=0.001,mu=0.0001), method="L-BFGS-B")
NIG.cf <- coef(NIG.fit)

#*************************************************************
#Functions and AIC
#*************************************************************
AIC(Norm.fit,gBm.fit,NIG.fit)

myfgBm <- function(u)
  dnorm(u, mean=gBm.cf["mu"], sd=gBm.cf["sigma"])
myfNorm <- function(u)
  dnorm(u, mean=Norm.cf["mu"],sd=Norm.cf["sigma"])
myfNIG <- function(u)
  dNIG(u, alpha=NIG.cf["alpha"],beta=NIG.cf["beta"],
       delta=NIG.cf["delta1"], mu=NIG.cf["mu"])
#*************************************************************
# EMPERICAL DENSITY COMPARED TO ESTIMATED DENSITTIES
#*************************************************************
# Empirical density
plot(density(Z, na.rm=TRUE), main="Empirical Density", col="black", lwd=2)
legend("topright", legend="Empirical", col="black", lty=1, cex=0.8, lwd=2)

# Comparing GBM model to Empirical density of returns
plot(density(Z, na.rm=TRUE), main="gBm versus Empirical", col="darkgrey", lwd=2)
curve(myfgBm, zMin, zMax, add=TRUE, col="darkorange", lty=1, lwd=2)
legend("topright", legend=c("Empirical", "gBm"), col=c("darkgrey", "darkorange"), lty=1, cex=1, lwd=2)

# Comparing Compound Poisson with Gaussian increments model to Empirical density of returns
plot(density(Z, na.rm=TRUE), main="Norm versus Empirical",col="darkgrey", lwd=2)
curve(myfNorm, zMin, zMax, add = TRUE, col="darkgreen", lty=1, lwd=2)
legend("topright",legend=c("Empirical", "Norm"), col=c("darkgrey", "darkgreen"), lty=1, cex = 0.8, lwd=2)

# Comparing Normal-Inverse Gaussian Lévy model to Empirical density of returns
plot(density(Z, na.rm=TRUE), main="NIG versus Empirical",col="darkgrey", lwd=2)
curve(myfNIG, zMin, zMax, add=TRUE, col="darkblue", lty=1, lwd=2)
legend("topright", legend=c("Empirical", "NIG"), col=c("darkgrey", "darkblue"), lty=1, cex=1, lwd=2)
#**********************************************************************
library(pracma)
# Set window size
window <- 100
# Function to calculate Hurst exponent in a moving window
hurstan <- function(x, window) {
  hurst <- rollapply(x, width=window, 
                     FUN=function(u) hurstexp(u, d = 50, display = TRUE)$He, by.column=FALSE)
  return(hurst)
}

# Apply function in a rolling window
hurst_exp <- hurstan(returns, window)
# Plot time-varying Hurst exponent
plot(hurst_exp, type="l", col="blue", ylab="Hurst Exponent", 
     main="Time-Varying Hurst Exponent of Ethereum Returns")
abline(h = 0.5, col="red")

#**********************************************************************************
# ORNSTEIN-UHLENBECK (OU) PROCESS WITH FRACTIONAL GAUSSIAN NOISE (WITH NO JUMPS)
#***********************************************************************************
modGbm <- setModel(drift="-mu*x", diffusion="sigma", hurst=NA)
yuima.obj <- setYuima(model=modGbm, data=setData(cumsum(Z1),delta=Dt))
#*******************************************************
#ESTIMATION OF PARAMETERS
#*******************************************************
qgv.est=qgv(yuima.obj) 
qgv.est
mmfrac.par=mmfrac(yuima.obj)
mmfrac.par
#*********************************************************************************
hurst.par=coef(mmfrac.par)[1]  
sigma=coef(mmfrac.par)[2]
mu=coef(mmfrac.par)[3]
#*************************************************************************************
set.seed(123)
true.params <- list(mu=coef(mmfrac.par)[3],sigma=coef(mmfrac.par)[2])
modsim <- setModel(drift="-mu*x", diffusion="sigma", hurst=hurst.par,
                   state.variable="x", time.variable ="t", solve.variable ="x",
                   xinit = Z1.0)
set.seed(1234)
grid <- setSampling(Terminal = 10,n=length(prices)-1)
L <- simulate(modsim, nsim=10, true.parameter = true.params,sampling = grid)
M <- L@data@original.data
#*****************************************************************************************
# LÉVY SIMULATION TO BE ADDED TO M
#******************************************************************************************
set.seed(1234)
mu <- 0 # mean of the process
theta <- NIG.cf["mu"] # mean reversion parameter
alpha <- NIG.cf["alpha"] # shape parameter of the normal inverse Gaussian distribution
beta <- NIG.cf["beta"] # skewness parameter of the normal inverse Gaussian distribution
delta <- NIG.cf["delta1"] # scale parameter of the normal inverse Gaussian distribution

#*****************************************************************************************
# DEFINE THE DELTARNG FUNCTION
#****************************************************************************************
deltarnig <- function(n, alpha, beta, delta, mu) {
  # Initialize an empty vector to store the random numbers
  x <- numeric(n)
  for (i in 1:n) {
    # Generate a random number from the uniform distribution on (0, 1)
    u <- runif(1)
    # Generate a random number from the standard normal distribution
    v <- rnorm(1)
    # Calculate w
    w <- alpha * sqrt(1 + v^2)
    # Calculate z
    z <- (beta + w) * v / (beta * w - 1)
    # Calculate x
    x[i] <- delta * z + mu
  }
  # Return the vector of random numbers
  return(x)
}
#******************************************************************************
#GENERATE A SAMPLE OF SIZE N FROM THE FOUL PROCESS
#*******************************************************************************
n <- length(prices)  # sample size
t <- seq(0, 10, length.out = n)  # time points
dt <- diff(t)   # time increments
x <- numeric(n)  # vector to store the sample
x[0] <- 0  # set the initial value
#***************************************************
#Combining the fraction OU with Levy process
#***************************************************
set.seed(1234)
X <- M + deltarnig(n, alpha = alpha, beta = beta,delta=delta,mu=mu)
length(t)
#Plot the sample path
plot(t, X, type = "l", main = "Sample path of FOUL process")
#*******************************************************************************
#GENERATE FEATURES FROM FRACTIONAL ORNSTEIN-UHLENBECK LÉVY PROCESS
#*******************************************************************************
# Volatility estimation
#*******************************************************************************
# Calculate MSE
X=X[1:length(returns)]
mse_FOUL_model <- mse(returns, X)
print(paste("MSE:", mse_FOUL_model))
# Calculate R-squared 
rsq_FOUL_model <- R2(returns, X)
print(paste("R-Squared:", rsq_FOUL_model)) 
# Calculate MAPE
mape_FOUL_model <- mape(returns, X) * 100 
print(paste("MAPE:", mape_FOUL_model))

# Realized volatility over daily intervals
daily_vols <- rollapply(X^2, width=1, sum)
vol <- sqrt(daily_vols)* 15
t=t[1:length(vol)]
# Plot the volatility series
plot(t, vol, type = "l",col="blue", main = "Volatility series of FOUL process")

#*********************************************************************************************
# LAGGED RETURNS
lag11 <- lag(returns, 1)
lag11 <- lag11[1:length(vol)]
#********************************************************************************************
# FEATURES
features <- na.omit(data.frame(lag1 = lag11, volatility = vol))

#********************************************************************************************
#SPLIT DATA INTO TRAINING AND TESTING SETS
trainIndex <- createDataPartition(returns, times=1, p=0.8, list=FALSE)
trainReturns <- returns[trainIndex]
trainFeatures <- features[trainIndex,]
trainFeatures <- na.roughfix(trainFeatures)
trainReturns <- na.roughfix(trainReturns)
testReturns <- returns[-trainIndex]
testFeatures <- features[-trainIndex,]
testFeatures <- na.roughfix(testFeatures)
testReturns <- na.roughfix(testReturns)
# Define hyperparameter grids
svm_grid <- expand.grid(C = c(0.1, 1, 10, 100), 
                        sigma = c(0.01, 0.1, 1))
# Train SVM model with grid search 
svm_model <- train(trainReturns, x = trainFeatures, 
                   method = "svmRadial", 
                   tuneGrid = svm_grid, 
                   trControl = trainControl(method = "cv", number = 5))
# Predict log returns on testing set
predReturns <- predict(svm_model, testFeatures)

#********************************************************************************
#REGRESSION METRICS/SVM MODEL PERFOMANCE
#********************************************************************************
library(Metrics)
# Subset to equal length
len <- min(length(testReturns), length(predReturns))
testReturns <- testReturns[1:len] 
predReturns <- predReturns[1:len]
# Calculate MSE
mse <- mse(testReturns, predReturns)
print(paste("MSE:", mse))

# Calculate R-squared 
rsq <- R2(testReturns, predReturns)
print(paste("R-Squared:", rsq)) 

# Calculate MAPE
mape <- mape(testReturns, predReturns) * 100 
print(paste("MAPE:", mape))


#********************************************************************************
#*SCATTER PLOT OF ACTUAL VS PREDICTED RETURNS
#*********************************************************************************
plot(testReturns, predReturns,
     main="Actual vs Predicted Returns",
     xlab="Actual Returns", ylab="Predicted Returns")
abline(0, 1, col="red")
#**************************************************************
