
## This file is to train DNN for estimating the optimal weight in point estimation for the
## Section 5.3 in the main article. One can further run the corresponding validation file
## to generate testing results with DNN locked in files

## Change working directory
setwd("~/Dropbox/Research/R_code_DNN_point_estimation/R_code_to_share_final/")
## Source functions
source("DNN_functions.r")
## Load required packages
library(mcmcplots)
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)

##############################################################################
set.seed(1) ## set random seed
n.first.itt = 10^3 ## training data size M
n.itt = 10^6 ## number of iterations to estimate optimal weight N

## simulate training data of \theta_1
DNN.first.pbo.rate.vec = runif(n.first.itt, min = 0.2, max = 0.7)
## simulate training data of \theta
DNN.first.delta.rate.vec = runif(n.first.itt, min = -0.2, max = 0.3)
## specify parameters for adaptive designs
n1 = 100; n2a = 50; n2b = 250; theta.cutoff = 0.16

n.cluster = 8 ## number of clusters in parallel computing
cl = makeCluster(n.cluster)
registerDoParallel(cl)
DNN.first.label = foreach(first.ind=1:n.first.itt) %dopar% {
  source("DNN_functions.r")
  library(mcmcplots)
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(tibble)

  ## set the underlying \theta_1, \theta, and \theta_2
  rate.pbo = DNN.first.pbo.rate.vec[first.ind]
  rate.delta = DNN.first.delta.rate.vec[first.ind]
  rate.trt = rate.pbo + rate.delta
  
  ## simulate training data
  DNN.first.train = t(sapply(1:n.itt, 
                             function(temp.ind){adaptive.bin.data.func(
                               rate.pbo.in = rate.pbo, 
                               rate.trt.in = rate.trt, 
                               theta.cutoff.in = theta.cutoff, 
                               n1.in = n1, 
                               n2a.in = n2a, 
                               n2b.in = n2b)}))
  
  ## calculate T1 and T2
  t1 = 0.5*DNN.first.train[, 1] + 0.5*DNN.first.train[, 2]
  t2 = DNN.first.train[, 1]
  
  first.w1 = mean((t2-t1)*(t2-rate.delta))/mean((t2-t1)^2)
  
  return(first.w1)

}
stopCluster(cl)

#######################################################################################
## cross-validation to select DNN structure
data.first.DNN.train =  as_tibble(as.matrix(cbind(DNN.first.pbo.rate.vec,
                                                  DNN.first.delta.rate.vec)))
data.first.DNN.train.scale =scale(data.first.DNN.train)
DNN.label.vec = unlist(DNN.first.label)

## standardize training data with mean zero and SD 1
col_means_first_train <- attr(data.first.DNN.train.scale, "scaled:center")
col_stddevs_first_train <- attr(data.first.DNN.train.scale, "scaled:scale")

## save scale parameters to a file
col_mean_sd = list("col_means_train" = col_means_first_train, 
                   "col_stddevs_train" = col_stddevs_first_train
)
save(col_mean_sd, file = "sim_3/sim_3_scale_parameters")

## candidate structures for DNN
DNN.cutoff.nodes.cross = c(40, 60, 40, 60)
DNN.cutoff.layers.cross = c(2, 2, 3, 3)
DNN.cross.mat = matrix(NA, nrow = length(DNN.cutoff.nodes.cross), ncol = 4)

## conduct cross validation with 20% as validation dataset
for (cross.ind in 1:length(DNN.cutoff.nodes.cross)){
  DNN.cal.cutoff.cross = DNN.fit(data.train.scale.in = data.first.DNN.train.scale,
                                 data.train.label.in = DNN.label.vec,
                                 drop.rate.in = 0.1,
                                 active.name.in = "relu",
                                 n.node.in = DNN.cutoff.nodes.cross[cross.ind],
                                 n.layer.in = DNN.cutoff.layers.cross[cross.ind],
                                 max.epoch.in = 10^3,
                                 batch_size_in = 100,
                                 validation.prop.in = 0.2,
                                 n.endpoint.in = 1)
  DNN.cross.mat[cross.ind, ] = 
    as.vector(unlist(lapply(DNN.cal.cutoff.cross$history$metrics,function(x){tail(x,1)})))
}

## select the one with minimum validation loss
DNN.cutoff.opt.nodes = DNN.cutoff.nodes.cross[which.min(DNN.cross.mat[, 4])]
DNN.cutoff.opt.layers = DNN.cutoff.layers.cross[which.min(DNN.cross.mat[, 4])]

## training final DNN with the selected structure
DNN.first.temp = DNN.fit(data.train.scale.in = data.first.DNN.train.scale,
                         data.train.label.in = DNN.label.vec,
                         drop.rate.in = 0.1,
                         active.name.in = "relu",
                         n.node.in = DNN.cutoff.opt.nodes,
                         n.layer.in = DNN.cutoff.opt.layers,
                         max.epoch.in = 10^3,
                         batch_size_in = 100,
                         validation.prop.in = 0,
                         n.endpoint.in = 1)
print(DNN.first.temp)
DNN.first.temp.weight = get_weights(DNN.first.temp$model)

## write trained DNN to files
save_model_hdf5(DNN.first.temp$model, 
                "sim_3/sim_3_DNN",
                overwrite = TRUE, include_optimizer = TRUE)





