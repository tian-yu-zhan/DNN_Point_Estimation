
## This file is to generate testing results in Table 1 and S1 of Section 5.1 with pre-training
## DNN obtained from the training file. The trained DNN and related files are also saved
## in the folder

## change working directory
setwd("~/Dropbox/Research/R_code_DNN_point_estimation/R_code_to_share_final/")
## source functions
source("DNN_functions.r")
## load packages
library(mcmcplots)
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)
library(HDCI)

####################################################################################
## generate testing results for two values of sample size
for (n.ind in 2){

set.seed(n.ind) ## random seed
if (n.ind==1) n.in = 2 ## sample size of 2
if (n.ind==2) n.in = 10 ## sample size of 10

n.test.itt = 10^6 ## validation iterations of 10^6

####################################################################################
## load DNN
DNN.first.model = load_model_hdf5(paste0("sim_1/sim_1_DNN_n_", n.in), 
                                   custom_objects = NULL, compile = TRUE)

###################################################################################
## load scale parameters
load(file = paste0("sim_1/sim_1_scale_parameters_n_", n.in))

col_means_first_train = col_mean_sd$col_means_train
col_stddevs_first_train = col_mean_sd$col_stddevs_train

####################################################################################
n.test.ind = 20 ## six scenarios for each given sample size

## output matrix
test.out.mat = matrix(NA, nrow =n.test.ind, ncol = 18)
colnames(test.out.mat) = c("theta", "k", 
                           "bias_mean", "bias_mle", "bias_rb", "bias_DNN", "bias_js",
                           "var_mean", "var_mle", "var_rb", "var_DNN", "var_js",
                           "MSE_mean", "MSE_mle", "MSE_rb", "MSE_DNN", "MSE_js",
                           "MSE_half")

for (test.ind in 1:n.test.ind){

  print(test.ind)
  ## set underlying parameters for theta and k
  if (test.ind<=10){
    test.k = 0.1
  } else{
    test.k = 0.9
  }
  
  test.theta = rep(seq(0.5, 5, length.out = 10),2)[test.ind]
 
  ## calculate \theta_RB, MLE and mean
  stats.12.mat = sapply(1:n.test.itt, function(itt){
    data.in = runif(n.in, min = (1-test.k)*test.theta, max = (1+test.k)*test.theta)

    stats.mle = theta_mle_correct(n = n.in, k = test.k, xn = max(data.in))
    stats.rb = theta_rb(x1 = min(data.in), xn = max(data.in))
    stats.js = theta_js(x_vec = data.in, k = test.k)
    
    stats.mean = mean(data.in)
    
    return(c(stats.mle, stats.rb, stats.mean, stats.js))
  })

  ## set T_1 as \theta_RB, T_2 as MLE
  stats.mle.vec = stats.12.mat[1, ]
  stats.1.vec = stats.rb.vec = stats.12.mat[2, ]
  stats.mean.vec = stats.12.mat[3, ]
  stats.2.vec = stats.js.vec = stats.12.mat[4, ]
  stats.12.half = (stats.1.vec + stats.2.vec)/2

  ## use T_1 as a plug-in estimator to estimate theta
  DNN.test.data.scale = scale(cbind(stats.1.vec, rep(test.k, n.test.itt)),
                    center = col_means_first_train, scale = col_stddevs_first_train)
  ## estimate optimal w from DNN 
  DNN.test.w1.pred = 
    DNN.first.model %>% predict(DNN.test.data.scale)
  ## compute ensemble estimator U
  DNN.stats.test = sapply(1:n.test.itt,
              function(temp.ind){sum(c(DNN.test.w1.pred[temp.ind], 
                      1-DNN.test.w1.pred[temp.ind])*
              c(stats.1.vec[temp.ind], stats.2.vec[temp.ind]))})
  
  ## write results to the output matrix
  test.out.mat[test.ind, ] = c(test.theta,
                               test.k, 
                               mean(stats.mean.vec) - test.theta,
                               mean(stats.mle.vec) - test.theta,
                               mean(stats.rb.vec) - test.theta,
                               mean(DNN.stats.test) - test.theta,
                               mean(stats.js.vec) - test.theta,
                               
                               var(stats.mean.vec),
                               var(stats.mle.vec),
                               var(stats.rb.vec),
                               var(DNN.stats.test),
                               var(stats.js.vec),
                               
                               mean((stats.mean.vec-test.theta)^2),
                               mean((stats.mle.vec-test.theta)^2),
                               mean((stats.rb.vec-test.theta)^2),
                               mean((DNN.stats.test-test.theta)^2),
                               mean((stats.js.vec-test.theta)^2),
                               mean((stats.12.half-test.theta)^2)
                               
                               )
  
  # print(test.out.mat)
}

test.out.mat = data.frame(test.out.mat)

## calculate relative efficiency with variance
test.out.mat$var_ratio_mean = test.out.mat$var_mean/test.out.mat$var_DNN
test.out.mat$var_ratio_mle = test.out.mat$var_mle/test.out.mat$var_DNN
test.out.mat$var_ratio_rb = test.out.mat$var_rb/test.out.mat$var_DNN
test.out.mat$var_ratio_js = test.out.mat$var_js/test.out.mat$var_DNN
## calculate relative efficiency with MSE
test.out.mat$MSE_ratio_mean = test.out.mat$MSE_mean/test.out.mat$MSE_DNN
test.out.mat$MSE_ratio_mle = test.out.mat$MSE_mle/test.out.mat$MSE_DNN
test.out.mat$MSE_ratio_rb = test.out.mat$MSE_rb/test.out.mat$MSE_DNN
test.out.mat$MSE_ratio_js = test.out.mat$MSE_js/test.out.mat$MSE_DNN
test.out.mat$MSE_ratio_half = test.out.mat$MSE_half/test.out.mat$MSE_DNN

## minimax studies
test.out.mat.temp = test.out.mat
test.out.mat.final= rbind(test.out.mat.temp[1:10,],
                     apply(test.out.mat.temp[1:10,], 2, min),
                     test.out.mat.temp[11:20,],
                     apply(test.out.mat.temp[11:20,], 2, min)
)
test.out.mat.final$theta[c(11,22)] = "min"

# print(test.out.mat.final)
## write results to files
write.csv(test.out.mat.final, paste0("sim_1/results_sim_1_n_",n.in,".csv"))

}







