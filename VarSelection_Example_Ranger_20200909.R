# load the relevant packages
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam","glmnet")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

#-----------------------------------------------------------------------------------------------------------------------------------------------
# A. SIMULATE DATA
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Simulate data set with 1000 observations
set.seed(123)
n=1000
sigma <- abs(matrix(runif(25,0,1), ncol=5))
sigma <- forceSymmetric(sigma)
sigma <- as.matrix(nearPD(sigma)$mat)
x <- rmvnorm(n, mean=c(0,.25,.15,0,.1), sigma=sigma)
modelMat<-model.matrix(as.formula(~ (x[,1]+x[,2]+x[,3]+x[,4]+x[,5])^3))
beta<-runif(ncol(modelMat)-1,0,1)
beta<-c(2,beta) # setting intercept
mu <- 1-plogis(modelMat%*%beta) # true underlying risk of the outcome
y<-rbinom(n,1,mu)

hist(mu);mean(y)

x<-data.frame(x)
D<-data.frame(x,y)

# Specify number of folds and create index for cross-validation
folds=5
index<-split(1:1000,1:folds)

# Split the data into 5 groups for 5-fold cross-validation
splt<-lapply(1:folds,function(ind) D[index[[ind]],])
# view the first 6 observations in the first [[1]] and second [[2]] folds
head(splt[[1]])
head(splt[[2]])

# screen.corRank from SuperLearner package
screen.corRank <- function (Y, X, family, method = "pearson", rank = 2, ...) 
{
  listp <- apply(X, 2, function(x, Y, method) {
    ifelse(var(x) <= 0, 1, cor.test(x, y = Y, method = method)$p.value)
  }, Y = Y, method = method)
  whichVariable <- (rank(listp) <= rank)
  return(whichVariable)
}



#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------
# B. FITTING SUPERLEARNER USING THE PACKAGE
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Once with  variable screening
set.seed(123)
ranger_learner = create.Learner("SL.ranger", params = list(num.trees = 500, mtry=2, min.node.size=10, replace=T))
sl.lib <- list("SL.mean", ranger_learner$names, "SL.glm",  c("SL.glm", "screen.corRank"))
SLfitY_scr<-SuperLearner(Y=y,X=x,family="binomial",
                         method="method.AUC",
                         SL.library=sl.lib,
                         cvControl=list(V=folds, validRows=index))
# Looking at coefficients, predictions, and which variables were selected
SLfitY_scr
SLfitY_scr$Z[,4]
SLfitY_scr$whichScreen
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# C. CODING SUPERLEARNER BY HAND
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# In my actual project, we are doing it this way to be able to include a custom algorithm 
# Here, we just want to make sure that what we're doing gives the same results as the SuperLearner function

# Fitting individual algorithms on the training set (minus the ii-th validation set)
# We had to use leave-one-out cross-validation for cv.glmnet, nfolds = 200, because the folds for cross-validation within each CV fold are chosen at random
set.seed(123)
m1 <- lapply(1:folds,function(ii) weighted.mean(rbindlist(splt[-ii])$y)) #mean - SL function uses weighted.mean
m2 <- lapply(1:folds, function(ii) ranger(y~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = T))
m3 <- lapply(1:folds, function(ii) glm(y~., data=do.call(rbind,splt[-ii]), family="binomial"))

# Writing separate functions for variable selection and cv.glmnet with var selection
vsfunc <- function(ii){

  whichVar <- screen.corRank(Y=do.call(rbind,splt[-ii])[,6], X=do.call(rbind,splt[-ii])[,-6])
  
  print(whichVar)

# This works!
  vars <- c(whichVar,TRUE)
  names(vars)[6] <- "y"
  
  m4 <- glm(y~., data=do.call(rbind,splt[-ii])[,vars], family="binomial")
  
  # You're only defining m4, not m4[ii]
  p4 <- predict(m4, newdata = do.call(rbind,splt[ii])[,vars], type="response")
  return(p4)
}
p4 <- lapply(1:folds, function(z) vsfunc(z))

# Use the model fits above to generate predictions in each fold
p1 <- lapply(1:folds, function(ii) rep(m1[[ii]],  nrow(splt[[ii]])))
p2 <- lapply(1:folds,function(ii) predict(m2[[ii]], data=do.call(rbind,splt[ii])))
p3 <- lapply(1:folds, function(ii) predict(m3[[ii]], newdata = do.call(rbind,splt[ii]), type="response"))

# Checking that what we did matches SL screening (predictions)
# With variable selection
cbind(sort(SLfitY_scr$Z[,4]),sort(as.numeric(do.call(rbind,p4))),round(sort(SLfitY_scr$Z[,4])-sort(as.numeric(do.call(rbind,p4))),4))
head(sort(SLfitY_scr$Z[,4]))
head(sort(as.numeric(do.call(rbind,p4))))

# Updating dataframe 'splt' so that column 1 has the observed outcome (y)
# and subsequent columns contain the predictions we generated above
for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,6], p1[[i]], p2[[i]]$predictions, p3[[i]], p4[[i]])
}
# Looking just at the first few observations in the first fold
head(data.frame(splt[[1]]))

# Generating CV risk estimates
risk1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for mean
risk2 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[i]][,1]))  # CV-risk for ranger
risk3 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,4], labels=splt[[i]][,1]))  # CV-risk for glm
risk4 <- lapply(1:folds, function(ii) 1-AUC(predictions=splt[[ii]][,5], labels=splt[[i]][,1]))  # CV-risk for glm screen

a<-rbind(cbind("mean",mean(do.call(rbind, risk1),na.rm=T)),
         cbind("ranger",mean(do.call(rbind, risk2),na.rm=T)),
         cbind("glm",mean(do.call(rbind,risk3), na.rm=T)),
         cbind("glm screen", mean(do.call(rbind,risk4), na.rm=T)))

# Also combine predicted probabilities for the metalearner
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","AC.mean","AC.ranger","AC.glm","AC.glmscreen")

# Define the function we want to optimize 
bounds = c(0, Inf)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("AC.mean","AC.ranger","AC.glm","AC.glmscreen")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}

# Optimization step
init <- rep(1/4, 4)
fit <- optim(par=init, fn=SL.r, A=X[,2:5], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])

# Check that convergence was achieved
fit

# Normalize coefficients and look at them
alpha<-fit$par/sum(fit$par)
alpha

# Compare output from SL function and hand-coded function 
SLfitY_scr
alpha
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
