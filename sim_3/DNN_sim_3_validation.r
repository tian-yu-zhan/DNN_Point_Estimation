
## This file is to generate testing results in Table 3, S3 and Figure S1 of Section 5.3 
## with pre-training DNN obtained from the training file. The trained DNN and related 
## files are also saved in the folder

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

##############################################################################
set.seed(1) ## set random seed
n.val.itt = 10^6 ## number of iterations in validation

## specify parameters in adaptive designs
n1 = 100; n2a = 50; n2b = 250; theta.cutoff = 0.16

####################################################################################
## load DNN
DNN.first.model = load_model_hdf5("sim_3/sim_3_DNN", 
                                  custom_objects = NULL, compile = TRUE)

###################################################################################
## load scale parameters
load(file = "sim_3/sim_3_scale_parameters")

col_means_first_train = col_mean_sd$col_means_train
col_stddevs_first_train = col_mean_sd$col_stddevs_train

################################################################################
## conduct validations
n.table.ind = 15 ## number of scenarios
critical.value.vec = seq(from=0.05, to=0.1, by = 0.001) ## candidate critical values
n.critical.value.vec = length(critical.value.vec)
table.out = matrix(NA, nrow = n.table.ind, ncol = 11+4*n.critical.value.vec)
colnames(table.out) = c("rate_p", "rate_t", "rate_delta",
                        "bias_1", "bias_2", "bias_3", 
                        "bias_DNN",
                        "MSE_1", "MSE_2", "MSE_3",
                        "MSE_DNN",
                        paste0("dec_1_", critical.value.vec),
                        paste0("dec_2_", critical.value.vec),
                        paste0("dec_3_", critical.value.vec),
                        paste0("dec_DNN_", critical.value.vec)
                        )

for (table.ind in 1:n.table.ind){
  
  print(table.ind)
  ## specify \theta_1 and \theta
  if (table.ind == 1) {rate.pbo.test = 0.42; rate.delta.test = 0}
  if (table.ind == 2) {rate.pbo.test = 0.47; rate.delta.test = 0}
  if (table.ind == 3) {rate.pbo.test = 0.52; rate.delta.test = 0}
  if (table.ind == 4) {rate.pbo.test = 0.66; rate.delta.test = 0}
  
  if (table.ind == 5) {rate.pbo.test = 0.42; rate.delta.test = 0.1}
  if (table.ind == 6) {rate.pbo.test = 0.42; rate.delta.test = 0.12}
  if (table.ind == 7) {rate.pbo.test = 0.42; rate.delta.test = 0.14}
  
  if (table.ind == 8) {rate.pbo.test = 0.47; rate.delta.test = 0.06}
  if (table.ind == 9) {rate.pbo.test = 0.47; rate.delta.test = 0.08}
  if (table.ind == 10) {rate.pbo.test = 0.47; rate.delta.test = 0.1}
  if (table.ind == 11) {rate.pbo.test = 0.47; rate.delta.test = 0.12}
  if (table.ind == 12) {rate.pbo.test = 0.47; rate.delta.test = 0.14}
  
  if (table.ind == 13) {rate.pbo.test = 0.52; rate.delta.test = 0.1}
  if (table.ind == 14) {rate.pbo.test = 0.52; rate.delta.test = 0.12}
  if (table.ind == 15) {rate.pbo.test = 0.52; rate.delta.test = 0.14}
  
  ## set \theta_2 from the treatment groups
  rate.trt.test = rate.pbo.test + rate.delta.test

  ## simulate validation data
  DNN.point.fit = t(sapply(1:n.val.itt, 
                           function(temp.ind){adaptive.bin.data.func(
                             rate.pbo.in = rate.pbo.test, 
                             rate.trt.in = rate.trt.test, 
                             theta.cutoff.in = theta.cutoff, 
                             n1.in = n1, 
                             n2a.in = n2a, 
                             n2b.in = n2b)}))
  
  ## obtain T2
  DNN.T2.est = DNN.point.fit[, 1]
  ## obtain \tilde{\theta}_1
  naive.1.est = DNN.point.fit[, 3]
  ## obtain \tilde{\theta}_2 and T1
  DNN.T1.est = naive.2.est = DNN.point.fit[, 4]
  ## obtain \tilde{\theta}_3
  naive.3.est = DNN.point.fit[, 5]
  ## obtain training data for DNN
  data.DNN.w1.est = cbind(DNN.point.fit[, 6]*0.5+DNN.point.fit[, 8]*0.5, naive.2.est)
  
  ## use T_1 as a plug-in estimator to estimate theta
  DNN.test.data.scale = scale(data.DNN.w1.est,
                              center = col_means_first_train, scale = col_stddevs_first_train)
  ## estimate optimal w from DNN 
  DNN.w1.est = 
    DNN.first.model %>% predict(DNN.test.data.scale)
  ## compute ensemble estimator U
  DNN.T1.T2.est = DNN.w1.est*DNN.T1.est + 
    (1-DNN.w1.est)*DNN.T2.est
  
  table.out[table.ind, ] = c(rate.pbo.test,
                             rate.trt.test,
                             rate.delta.test,
                             ## bias
                             mean(naive.1.est)-rate.delta.test,
                             mean(naive.2.est)-rate.delta.test,
                             mean(naive.3.est)-rate.delta.test,
                             mean(DNN.T1.T2.est)-rate.delta.test,
                             ## variance
                             var(naive.1.est) + (mean(naive.1.est)-rate.delta.test)^2,
                             var(naive.2.est) + (mean(naive.2.est)-rate.delta.test)^2,
                             var(naive.3.est) + (mean(naive.3.est)-rate.delta.test)^2,
                             var(DNN.T1.T2.est) + (mean(DNN.T1.T2.est)-rate.delta.test)^2,
                             ## compute type I error rate / power
                             sapply(1:n.critical.value.vec, 
                                    function(temp){mean(naive.1.est>=
                                                          critical.value.vec[temp])}),
                             sapply(1:n.critical.value.vec, 
                                    function(temp){mean(naive.2.est>=
                                                          critical.value.vec[temp])}),
                             sapply(1:n.critical.value.vec, 
                                    function(temp){mean(naive.3.est>=
                                                          critical.value.vec[temp])}),
                             sapply(1:n.critical.value.vec, 
                                    function(temp){mean(DNN.T1.T2.est>=
                                                          critical.value.vec[temp])})
                             
  )
  
}
table.out = data.frame(table.out)

## compute relative efficiency
table.out$MSE_ratio_1_DNN = table.out$MSE_1/table.out$MSE_DNN
table.out$MSE_ratio_2_DNN = table.out$MSE_2/table.out$MSE_DNN
table.out$MSE_ratio_3_DNN = table.out$MSE_3/table.out$MSE_DNN

print(table.out$MSE_ratio_1_DNN)
print(table.out$MSE_ratio_2_DNN)
print(table.out$MSE_ratio_3_DNN)

## write results to files
write.csv(table.out, paste0("sim_3/results_sim_3_n1_",
                            n1,
                            "_n2a_", n2a,
                            "_n2b_", n2b,
                            "_theta_", theta.cutoff,
                            "_results.csv"))

# ## 0.064, 0.068, 0.094, 0.064
# print(table.out$dec_1_0.064)
# print(table.out$dec_2_0.068)
# print(table.out$dec_3_0.094)
# print(table.out$dec_DNN_0.064)

test.out = cbind(table.out$dec_1_0.064,
                 table.out$dec_2_0.068,
                 table.out$dec_3_0.094,
                 table.out$dec_DNN_0.064
                 )
















