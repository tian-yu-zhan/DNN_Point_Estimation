
## This file is to generate testing results in Table 2 and S2 of Section 5.2 with pre-training
## DNN obtained from the training file. The trained DNN and related files are also saved
## in the folder

## change working directory
setwd("~/Dropbox/Research/R_code_DNN_point_estimation/R_code_to_share_final/")
## source functions
source("DNN_functions.r")
## load packages
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)
library(rootSolve)
library(limSolve)
library(HDCI)
library(glmnet)
library(quantreg)

##############################################################################
## specify parameters
n.val.itt = 10^5 ## number of iterations for validation
n.cluster = 8 ## number of clusters for parallel computing

####################################################################################
## load DNN
DNN.first.model = load_model_hdf5("sim_2_qr/sim_2_DNN", 
                                  custom_objects = NULL, compile = TRUE)

###################################################################################
## load scale parameters
load(file = "sim_2_qr/sim_2_scale_parameters")

col_means_first_train = col_mean_sd$col_means_train
col_stddevs_first_train = col_mean_sd$col_stddevs_train
x.obs = col_mean_sd$x_obs ## observed data x
n.coef = dim(x.obs)[2] ## number of covariates including the intercept
n.obs = dim(x.obs)[1] ## number of observations

################################################################################
## conduct validations
n.table.ind = 9 ## number of validation scenarios
## output matrix
table.out = matrix(NA, nrow = n.table.ind, ncol = 13*n.coef)
colnames(table.out) = c(paste0("beta_", 1:n.coef),
                        paste0("DNN_bias_", 1:n.coef),
                        paste0("wlse_bias_", 1:n.coef),
                        paste0("lse_bias_", 1:n.coef),
                        paste0("lasso_bias_", 1:n.coef),
                        paste0("ridge_bias_", 1:n.coef),
                        paste0("qr_bias_", 1:n.coef),
                        paste0("MSE_DNN_", 1:n.coef),
                        paste0("MSE_wlse_", 1:n.coef),
                        paste0("MSE_lse_", 1:n.coef),
                        paste0("MSE_lasso_", 1:n.coef),
                        paste0("MSE_ridge_", 1:n.coef),
                        paste0("MSE_qr_", 1:n.coef)
)
table.out = data.frame(table.out)
# table.y.out = matrix(NA, nrow = n.table.ind, ncol = 10+n.coef)
# colnames(table.y.out) = c(paste0("beta_", 1:n.coef),
#                           "Bias_y_DNN", "Bias_y_wlse", "Bias_y_lse", "Bias_y_lasso",
#                           "Bias_y_ghat",
#                           "MSE_y_DNN", "MSE_y_wlse", "MSE_y_lse", "MSE_y_lasso",
#                           "MSE_y_ghat"
#                           )
# table.y.out = data.frame(table.y.out)

for (table.ind in 1:n.table.ind){
  
  ## assign underlying validation parameters
  if (table.ind==1) beta.vec.val = c(1, 1, 1, -1)*0.2
  if (table.ind==2) beta.vec.val = c(1, 1, -1, -1)*0.2
  if (table.ind==3) beta.vec.val = c(1, -1, -1, -1)*0.2
  
  if (table.ind==4) beta.vec.val = c(1, 1, 1, -1)*0.6
  if (table.ind==5) beta.vec.val = c(1, 1, -1, -1)*0.6
  if (table.ind==6) beta.vec.val = c(1, -1, -1, -1)*0.6
  
  if (table.ind==7) beta.vec.val = c(1, 1, 1, -1)*1.2
  if (table.ind==8) beta.vec.val = c(1, 1, -1, -1)*1.2
  if (table.ind==9) beta.vec.val = c(1, -1, -1, -1)*1.2
  
  ## simulate validation data
  cl = makeCluster(n.cluster)
  registerDoParallel(cl)
  DNN.val.fit.vec = foreach(ind.val.itt=1:n.val.itt) %dopar% {
    ## source functions
    source("DNN_functions.r")
    ## load packages
    library(keras)
    library(reticulate)
    library(tensorflow)
    library(tibble)
    library(doParallel)
    library(rootSolve)
    library(limSolve)
    library(HDCI)
    library(glmnet)
    library(quantreg)
    
    set.seed(ind.val.itt + n.val.itt*table.ind)  ## random seed
    
    return(reg.data.func(n.obs.in = n.obs, 
                              beta.vec.in = beta.vec.val, 
                              a.in = 1, 
                              MLE.in = FALSE,
                              comp.in = TRUE))
  }
  stopCluster(cl)
  
  DNN.val.fit = matrix(unlist(DNN.val.fit.vec), nrow = n.val.itt, ncol = 6*n.coef,
                       byrow = TRUE)

  # DNN.val.fit = t(sapply(1:n.val.itt, 
  #                        function(ind.val.itt){
  #                          reg.data.func(n.obs.in = n.obs, 
  #                                        beta.vec.in = beta.vec.val, 
  #                                        a.in = 1, 
  #                                        MLE.in = FALSE,
  #                                        comp.in = TRUE)
  #                        }))

  ## obtain T_1 and T_2
  wlse.point.val = DNN.val.fit[, 1:n.coef]
  lse.point.val = DNN.val.fit[, (1+n.coef):(2*n.coef)] 
  # mle.point.val = DNN.val.fit[, (1+2*n.coef):(3*n.coef)] 
  lasso.point.val = DNN.val.fit[, (1+3*n.coef):(4*n.coef)] 
  ridge.point.val = DNN.val.fit[, (1+4*n.coef):(5*n.coef)] 
  qr.point.val = DNN.val.fit[, (1+5*n.coef):(6*n.coef)] 

  ## use T_1 as a plug-in estimator to estimate theta
  DNN.test.data.scale = scale(qr.point.val,
                              center = col_means_first_train, 
                              scale = col_stddevs_first_train)
  ## estimate optimal w from DNN 
  DNN.w1.fit = 
    DNN.first.model %>% predict(DNN.test.data.scale)

  ## calculate ensemble estimator U
  DNN.point.val = t(sapply(1:n.val.itt, function(val.itt){
    t1.temp = qr.point.val[val.itt, ]
    t2.temp = wlse.point.val[val.itt, ]
    w1.temp = as.numeric(DNN.w1.fit[val.itt, ])
    DNN.temp = t1.temp*w1.temp + t2.temp*(1-w1.temp)
    return(DNN.temp)
  }))

  ## 
  # DNN.y.fit = DNN.point.val%*%t(x.obs)
  # DNN.y.fit = apply(DNN.y.fit, 1, mean)
  # 
  # lm.w.y.fit = DNN.val.fit[, 17]
  # lm.y.fit = DNN.val.fit[, 18]
  # lasso.y.fit = DNN.val.fit[, 19]
  # ghat.y.fit = DNN.val.fit[, 20]
  # 
  # y.true = mean(x.obs%*%as.matrix(beta.vec.val))
  
  ## write results to the output of y estimates
  # table.y.out[table.ind, ] = c(beta.vec.val, 
  #                                (mean(DNN.y.fit)-y.true),
  #                                (mean(lm.w.y.fit)-y.true),
  #                                (mean(lm.y.fit)-y.true),
  #                                (mean(lasso.y.fit)-y.true),
  #                                (mean(ghat.y.fit)-y.true),
  #                                
  #                                (mean((DNN.y.fit-y.true)^2)),
  #                                (mean((lm.w.y.fit-y.true)^2)),
  #                                (mean((lm.y.fit-y.true)^2)),
  #                                (mean((lasso.y.fit-y.true)^2)),
  #                                (mean((ghat.y.fit-y.true)^2)))
  
  ## write results to the output matrix of beta estimates
  table.out[table.ind, ] = c(beta.vec.val,
                             apply(DNN.point.val, 2, mean)-beta.vec.val,
                             apply(wlse.point.val, 2, mean)-beta.vec.val,
                             apply(lse.point.val, 2, mean)-beta.vec.val,
                             apply(lasso.point.val, 2, mean)-beta.vec.val,
                             apply(ridge.point.val, 2, mean)-beta.vec.val,
                             apply(qr.point.val, 2, mean)-beta.vec.val,

                             apply(DNN.point.val, 2, var)+
                               (apply(DNN.point.val, 2, mean)-beta.vec.val)^2,
                             apply(wlse.point.val, 2, var)+
                               (apply(wlse.point.val, 2, mean)-beta.vec.val)^2,
                             apply(lse.point.val, 2, var)+
                               (apply(lse.point.val, 2, mean)-beta.vec.val)^2,
                             apply(lasso.point.val, 2, var)+
                               (apply(lasso.point.val, 2, mean)-beta.vec.val)^2,
                             apply(ridge.point.val, 2, var)+
                               (apply(ridge.point.val, 2, mean)-beta.vec.val)^2,
                             apply(qr.point.val, 2, var)+
                               (apply(qr.point.val, 2, mean)-beta.vec.val)^2
  )
  if (sum(is.na(table.out[table.ind, ]))>0) stop("b")
  print(table.out)
}

table.out$MSE_DNN_total = table.out$MSE_DNN_1 + table.out$MSE_DNN_2 + 
  table.out$MSE_DNN_3 + table.out$MSE_DNN_3
table.out$MSE_wlse_total = table.out$MSE_wlse_1 + table.out$MSE_wlse_2 + 
  table.out$MSE_wlse_3 + table.out$MSE_wlse_4 
table.out$MSE_lse_total = table.out$MSE_lse_1 + table.out$MSE_lse_2 + 
  table.out$MSE_lse_3 + table.out$MSE_lse_4 
table.out$MSE_lasso_total = table.out$MSE_lasso_1 + table.out$MSE_lasso_2 + 
  table.out$MSE_lasso_3 + table.out$MSE_lasso_4
table.out$MSE_ridge_total = table.out$MSE_ridge_1 + table.out$MSE_ridge_2 + 
  table.out$MSE_ridge_3 + table.out$MSE_ridge_4
table.out$MSE_qr_total = table.out$MSE_qr_1 + table.out$MSE_qr_2 + 
  table.out$MSE_qr_3 + table.out$MSE_qr_4

print(table.out)

write.csv(table.out, "sim_2_qr/results_sim_2.csv")





