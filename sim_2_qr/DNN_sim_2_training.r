
## This file is to train DNN for estimating the optimal weight in point estimation for the
## Section 5.2 in the main article. One can further run the corresponding validation file
## to generate testing results with DNN locked in files

## Change working directory
setwd("~/Dropbox/Research/R_code_DNN_point_estimation/R_code_to_share_final/")
## Load required packages
source("DNN_functions.r")
## source functions
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)
library(rootSolve)
library(limSolve)
library(quantreg)

##############################################################################
## specify parameters
set.seed(1) ## set random seed
n.first.itt = 10^3 ## number of training dataset M
n.itt = 10^5 ## number of iterations N when estimating optimal weight
n.cluster = 8 ## number of cluster for parallel computing

## simulate underlying beta parameters
DNN.first.beta.1.vec = runif(n.first.itt, min = -1.5, max = 1.5)
DNN.first.beta.2.vec = runif(n.first.itt, min = -1.5, max = 1.5)
DNN.first.beta.3.vec = runif(n.first.itt, min = -1.5, max = 1.5)
DNN.first.beta.4.vec = runif(n.first.itt, min = -1.5, max = 1.5)

n.obs = 100 # number of observations
n.coef = 4 # number of covariates including the intercept
x.coef = matrix(runif(n.obs*(n.coef-1), min=-2, max = 2),
                nrow = n.obs, ncol = (n.coef-1)) ## fixed data of x
x.obs = cbind(rep(1, n.obs), x.coef) ## add intercept

###################################################################
## parallel computing to generate training data for DNN
cl = makeCluster(n.cluster)
registerDoParallel(cl)
DNN.first.label = foreach(first.ind=1:n.first.itt) %dopar% {
  source("DNN_functions.r")
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(tibble)
  library(rootSolve)
  library(limSolve)
  library(quantreg)
  
  ## assign underlying beta
  beta.1 = DNN.first.beta.1.vec[first.ind]
  beta.2 = DNN.first.beta.2.vec[first.ind]
  beta.3 = DNN.first.beta.3.vec[first.ind]
  beta.4 = DNN.first.beta.4.vec[first.ind]

  ## beta vector
  beta.train.vec = c(beta.1, beta.2, beta.3, beta.4)

  ## simulate training data
  DNN.first.train = t(sapply(1:n.itt, 
                             function(ind.itt){
                               reg.data.func(n.obs.in = n.obs, 
                                             beta.vec.in = beta.train.vec, 
                                             a.in = 1, 
                                             MLE.in = FALSE,
                                             comp.in = FALSE)
                             }))
  
  ## T_1 as quantile regression, and T_2 as the weighted LSE
  t1.beta.vec = DNN.first.train[, (1+5*n.coef):(6*n.coef)]
  t2.beta.vec = DNN.first.train[, (1+n.coef)]
  
  
  ## compute optimal weights
  w1.num = apply((t2.beta.vec-t1.beta.vec)*(t2.beta.vec-
        matrix(rep(beta.train.vec, n.itt), nrow = n.itt, ncol=n.coef, byrow=TRUE)), 
        2, 
                 function(temp){mean(temp, na.rm=TRUE)})
  w1.den = apply((t2.beta.vec-t1.beta.vec)^2, 2, 
                 function(temp){mean(temp, na.rm=TRUE)})
  w1.vec = w1.num / w1.den
  
  return(w1.vec)
  
}
stopCluster(cl)

## output training labels for DNN
DNN.first.label.mat = matrix(unlist(DNN.first.label), nrow = n.first.itt, ncol = n.coef,
                             byrow = TRUE)
print(DNN.first.label.mat)

#######################################################################################
## train the DNN
data.first.DNN.train =  as_tibble(as.matrix(cbind(DNN.first.beta.1.vec,
                                                  DNN.first.beta.2.vec,
                                                  DNN.first.beta.3.vec,
                                                  DNN.first.beta.4.vec
                                                  )))
data.first.DNN.train.scale =scale(data.first.DNN.train)

## standardize data to mean zero and SD 1
col_means_first_train <- attr(data.first.DNN.train.scale, "scaled:center")
col_stddevs_first_train <- attr(data.first.DNN.train.scale, "scaled:scale")

## save scale parameters and observed data x to a file
col_mean_sd = list("col_means_train" = col_means_first_train, 
                   "col_stddevs_train" = col_stddevs_first_train,
                   "x_obs" = x.obs
)
save(col_mean_sd, file = "sim_2_qr/sim_2_scale_parameters")

## candidate structures for DNN
DNN.cutoff.nodes.cross = c(40, 60, 40, 60)
DNN.cutoff.layers.cross = c(2, 2, 3, 3)
DNN.cross.mat = matrix(NA, nrow = length(DNN.cutoff.nodes.cross), ncol = 4)

## conduct cross validation with 20% as validation dataset
for (cross.ind in 1:length(DNN.cutoff.nodes.cross)){
  DNN.cal.cutoff.cross = DNN.fit(data.train.scale.in = data.first.DNN.train.scale,
                                 data.train.label.in = DNN.first.label.mat,
                                 drop.rate.in = 0.1,
                                 active.name.in = "relu",
                                 n.node.in = DNN.cutoff.nodes.cross[cross.ind],
                                 n.layer.in = DNN.cutoff.layers.cross[cross.ind],
                                 max.epoch.in = 10^3,
                                 batch_size_in = 100,
                                 validation.prop.in = 0.2,
                                 n.endpoint.in = n.coef)
  DNN.cross.mat[cross.ind, ] = 
    as.vector(unlist(lapply(DNN.cal.cutoff.cross$history$metrics,function(x){tail(x,1)})))
}

## select the one with minimum validation loss
DNN.cutoff.opt.nodes = DNN.cutoff.nodes.cross[which.min(DNN.cross.mat[, 4])]
DNN.cutoff.opt.layers = DNN.cutoff.layers.cross[which.min(DNN.cross.mat[, 4])]

## training final DNN with the selected structure
DNN.first.temp = DNN.fit(data.train.scale.in = data.first.DNN.train.scale,
                         data.train.label.in = DNN.first.label.mat,
                         drop.rate.in = 0.1,
                         active.name.in = "relu",
                         n.node.in = DNN.cutoff.opt.nodes,
                         n.layer.in = DNN.cutoff.opt.layers,
                         max.epoch.in = 10^3,
                         batch_size_in = 100,
                         validation.prop.in = 0,
                         n.endpoint.in = n.coef)
print(DNN.first.temp)
DNN.first.temp.weight = get_weights(DNN.first.temp$model)

## write trained DNN to files
save_model_hdf5(DNN.first.temp$model, 
                "sim_2_qr/sim_2_DNN", 
                overwrite = TRUE, include_optimizer = TRUE)






