
## This file is to train DNN for estimating the optimal weight in point estimation for the
## Section 5.1 in the main article. One can further run the corresponding validation file
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

##################################################################################
## Simulate training data for the values of sample size
for (n.ind in 2){
  
set.seed(n.ind) ## set random seed
if (n.ind==1) n.in = 2 ## sample size at 2
if (n.ind==2) n.in = 10 ## sample size at 10

n.itt = 10^6 ## number of iterations N when estimating optimal weight
n.train.itt = 10^3 ## number of training data size M

## simulate theta from certain range
DNN.theta.vec = runif(n = n.train.itt, min = 0.2, max = 10)
## training data of known design parameters k
DNN.k.vec = rep(c(0.1, 0.9), length.out = n.train.itt)
## training label for DNN
DNN.label.vec = rep(NA, n.train.itt)

for (train.itt in 1:n.train.itt){
  
  print(train.itt)
  ## underlying parameter theta
  true.theta = DNN.theta.vec[train.itt]
  ## known design paramter
  true.k = DNN.k.vec[train.itt]
  
  ## calcualte T_1 as \theta_RB and T_2 as MLE
  stats.12.mat = sapply(1:n.itt, function(itt){
    data.in = runif(n.in, min = (1-true.k)*true.theta, max = (1+true.k)*true.theta)
    
    stats.1 = theta_rb(x1 = min(data.in), xn = max(data.in))
    stats.2 = theta_js(x_vec = data.in, k = true.k)
    # stats.2 = theta_mle_correct(n = n.in, k = true.k, xn = max(data.in))

    return(c(stats.1, stats.2))
  })
  
  stats.1.vec = stats.12.mat[1, ]
  stats.2.vec = stats.12.mat[2, ]
  
  ## compute the estimated optimal weight
  w1 = mean((stats.2.vec-stats.1.vec)*(stats.2.vec-true.theta))/
    mean((stats.2.vec-stats.1.vec)^2)
  
  print(w1)
  
  DNN.label.vec[train.itt] = w1
}

#######################################################################################
## cross validation to select a proper DNN structure
data.first.DNN.train =  as_tibble((as.matrix(cbind(DNN.theta.vec, DNN.k.vec))))
data.first.DNN.train.scale =scale(data.first.DNN.train)

## normalize training data with mean zero and standard deviation 1
col_means_first_train <- attr(data.first.DNN.train.scale, "scaled:center")
col_stddevs_first_train <- attr(data.first.DNN.train.scale, "scaled:scale")

## save scale parameters to a file
col_mean_sd = list("col_means_train" = col_means_first_train, 
                       "col_stddevs_train" = col_stddevs_first_train
)
save(col_mean_sd, file = paste0("sim_1/sim_1_scale_parameters_n_", n.in))

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
                paste0("sim_1/sim_1_DNN_n_", n.in), 
                overwrite = TRUE, include_optimizer = TRUE)

}







