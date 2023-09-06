
ME <- function( y_true, y_pred ) {
  K <- backend()
  # K$square(K$mean( y_true -y_pred))
  K$mean(K$square( y_true -y_pred)) 
}

###################################################################################################################
## functions
stat.1.func = function(data.in){
  mean(data.in)
}

stat.2.func = function(data.in){
  s = sqrt(var(data.in)*(n-1)/n)
  cn = min(sqrt(n/2/a), sqrt(n)*gamma((n-1)/2)/sqrt(2*a)/gamma(n/2), na.rm=TRUE)
  s*cn
}

stat.3.func = function(data.in){
  x.abs = abs(mean(data.in))
  lambda.abs = sqrt(n/a)
  beta.abs = 1/(2*pnorm(lambda.abs)-1+sqrt(2/pi)/lambda.abs*exp(-lambda.abs^2/2))
  
  newlist = list("stat" = x.abs*beta.abs, "beta.abs" = beta.abs)
  newlist
}

stat.12.func = function(stat.1.in, stat.2.in){
  v1 = a/n
  v2 = min((n-1)/2-1, (n-1)/2*(gamma((n-1)/2))^2/gamma(n/2)^2-1, na.rm = TRUE)
  w12 = v2/(v1+v2)
  opt.12 = stat.1.in*w12 + stat.2.in*(1-w12)
  return(opt.12)
}

stat.23.func = function(beta.abs.in, stat.2.in, stat.3.in){
  v3 = (beta.abs.in^2-1)+a*beta.abs.in^2/n
  v2 = min((n-1)/2-1, (n-1)/2*(gamma((n-1)/2))^2/gamma(n/2)^2-1, na.rm = TRUE)
  w23 = v2/(v3+v2)
  opt.23 = stat.3.in*w23 + stat.2.in*(1-w23)
  return(opt.23)
}

data.sim.func = function(theta.func, n.func){
  data.sim.table = sapply(1:n.func, function(func.itt){
    
    data.sample = rnorm(n, mean = theta.func, sd = sqrt(a)*theta.func)  
    stat.1 = stat.1.func(data.sample)
    stat.2 = stat.2.func(data.sample)
    
    stat.3.fit = stat.3.func(data.sample)
    stat.3 = stat.3.fit$stat
    
    ## w12
    stat.12 = stat.12.func(stat.1, stat.2)
    
    ## w23
    stat.23 = stat.23.func(stat.3.fit$beta.abs, stat.2, stat.3)
    
    c(stat.1, stat.2, stat.12, stat.3, stat.23)
  })
}

stats.func = function(t1.func, t2.func){
  
  n.func = length(t1.func)
  t1.t2.func = (t1.func+t2.func)/2
  DNN.data.func.scale = scale(t1.t2.func,
                              center = col_means_first_train, scale = col_stddevs_first_train)
  
  ## iterative part to stabilize results
  algo.ind = TRUE
  # DNN.test.w1.pred.init = DNN.first.temp$model %>% predict(DNN.data.func.scale)
  
  # DNN.test.w1.pred.init = rep(0.1, n.func)
  
  DNN.test.w1.pred.init = pred.DNN.normal(DNN.final.weights.in = DNN.first.weight,
                                          n.layer.in = 2,
                                          data.train.in = (as.matrix(t1.t2.func)),
                                          col_means_train.in = col_means_first_train,
                                          col_stddevs_train.in = col_stddevs_first_train)
  algo.itt = 1
  while ((algo.ind)&(algo.itt<=algo.max)){
    DNN.stats.test = sapply(1:n.func,
                            function(temp.ind){sum(c(DNN.test.w1.pred.init[temp.ind], 
                                                     1-DNN.test.w1.pred.init[temp.ind])*
                                                     c(t1.func[temp.ind], t2.func[temp.ind]))})
    # DNN.test.data.scale = scale(DNN.stats.test,
    #                             center = col_means_first_train, scale = col_stddevs_first_train)
    # DNN.test.w1.pred = DNN.first.temp$model %>% predict(DNN.test.data.scale)
    
    # DNN.test.w1.pred = rep(0.1, n.func)
    
    DNN.test.w1.pred = pred.DNN.normal(DNN.final.weights.in = DNN.first.weight,
                                       n.layer.in = 2,
                                       data.train.in = (as.matrix(DNN.stats.test)),
                                       col_means_train.in = col_means_first_train,
                                       col_stddevs_train.in = col_stddevs_first_train)
    
    if (max(abs(DNN.test.w1.pred-DNN.test.w1.pred.init))<=algo.tol) algo.ind = FALSE
    print(DNN.stats.test)
    DNN.test.w1.pred.init = DNN.test.w1.pred
    algo.itt = algo.itt + 1
  }
  # print(algo.itt)
  return(DNN.stats.test)
  
}

## functions of example 1

theta_rb = function(x1, xn){
  (x1+xn)/2
}

theta_mle_correct = function(n, k, xn){
  1/(1+(n-1)/(n+1)*k)*xn
}

theta_LV = function(n, k, x1, xn){
  1/(2*(k^2*(n-1)/(n+1)+1))*((1-k)*x1+(1+k)*xn)
}

theta_bayes = function(n, k, a, x1, xn){
  h = x1/(1-k)
  l = xn/(1+k)
  (n-1+a)/(n-2+a)*(1-(h/l-1)/((h/l)^(n-1+a)-1))*l
}

theta_mean = function(x_vec){
  mean(x_vec)
}

theta_js = function(x_vec, k){
  m.temp = 3
  mle.temp = max(x_vec)/(1+(length(x_vec)-1)/(length(x_vec)+1)*k)
  js.stats.temp = (1-(m.temp-2)*
            (mle.temp^2*k^2/3)/length(x_vec)/m.temp/mle.temp^2)*mle.temp
  return(js.stats.temp)
}


##### example 2 functions
reg.data.single.variable.func = function(n.obs.in, beta.0.in, beta.1.in, a.in, MLE.in){
  
  ## simulate data
  x.beta.vec = beta.0.in + beta.1.in*x.obs
  y.obs = rnorm(n.obs.in, mean = x.beta.vec, sd = sqrt(a.in*x.beta.vec^2))
  
  data.in = data.frame("x1" = x.obs, "y" = y.obs)
  
  ## OLS
  lm.fit = lm(y~1+x1, data = data.in)
  lm.est = lm.fit$coefficients
  
  ## WOLS
  data.w = data.in
  data.w$w = pmin(1/(lm.est[1] + lm.est[2]*x.obs)^2, 10^5)
  
  lm.w.fit = lm(y~1+x1, data = data.w, weights = w)
  lm.w.est = lm.w.fit$coefficients
  
  ## MLE
  if (MLE.in){
    x.mat = matrix(cbind(rep(1, n.obs), x.obs), nrow = n.obs, ncol = 2)
    
    solve.function = function(temp.est){
      xtbeta = x.mat%*%temp.est
      obj.1 = sapply(1:n.obs, function(ind.obs){
        -x.mat[ind.obs,]/xtbeta[ind.obs]
      })
      obj.2 = sapply(1:n.obs, function(ind.obs){
        (y.obs[ind.obs] - xtbeta[ind.obs])/
          ((xtbeta[ind.obs])^3)*y.obs[ind.obs]*x.mat[ind.obs,]/a.in
      })
      obj.12 = apply(obj.1, 1, sum) + apply(obj.2, 1, sum)
      return(obj.12)
    }
    
    solve.I.function = function(temp.est){
      xtbeta = x.mat%*%temp.est
      obj.1 = sapply(1:n.obs, function(ind.obs){
        -as.matrix(x.mat[ind.obs,])%*%t(as.matrix(x.mat[ind.obs,]))/((xtbeta[ind.obs])^2)
      })
      obj.2 = sapply(1:n.obs, function(ind.obs){
        (-3*y.obs[ind.obs] +2*xtbeta[ind.obs])/
          ((xtbeta[ind.obs])^4)*y.obs[ind.obs]/a.in*
          as.matrix(x.mat[ind.obs,])%*%t(as.matrix(x.mat[ind.obs,]))
      })
      obj.12 = apply(obj.1, 1, sum) + apply(obj.2, 1, sum)
      obj.12.mat = matrix(obj.12, nrow=2,ncol=2,byrow = TRUE)
      return(obj.12.mat)
    }
    
    tol = 10^(-5)
    newton.ind = TRUE
    newton.count = 1
    est.init = c(0.1, 0.1)
    while(newton.ind){
      est.up = est.init - limSolve::Solve(solve.I.function(est.init))%*%
        as.matrix(solve.function(est.init))
      if (max(abs(est.up-est.init))<tol) newton.ind = FALSE
      if (newton.count>100) newton.ind = FALSE
      est.init = est.up
      newton.count = newton.count + 1
    }
    
    mse.est = est.up
    # mse.est = multiroot(f = solve.function, start = c(0.1, 0.1))$root
  } else{
    mse.est = c(0, 0)
  }
  
  return.list = c(lm.w.est, lm.est, mse.est)
  # print(return.list)
  
  return(return.list)
  
}

reg.data.func = function(n.obs.in, beta.vec.in, a.in, MLE.in, comp.in){
  
  ## simulate data
  x.beta.vec = as.numeric(x.obs%*%as.matrix(beta.vec.in))
  y.obs = rnorm(n.obs.in, mean = x.beta.vec, sd = sqrt(a.in*x.beta.vec^2))
  # y.obs = rnorm(n.obs.in, mean = x.beta.vec, sd = 1)
  
  data.in = cbind(x.obs, y.obs)
  colnames(data.in) = c( paste0("x", 1:n.coef), "y")
  data.in = data.frame(data.in)
  
  if (comp.in){
    
    ## kernel regression
    # bw = npregbw(y~x2+x3+x4, data = data.in)
    # ghat = npreg(bw)
    # ghat.y.est = mean(ghat$mean)
    # ghat.y.est = 0
    
    ## Ridge regression
    cv_ridge = cv.glmnet(as.matrix(data.in[,c(2:n.coef)]), data.in$y, alpha = 0, 
                         lambda = 10^seq(2, -3, length.out=10), data = data.in)
    opt.lambda = cv_ridge$lambda.min
    ridge.fit = glmnet(as.matrix(data.in[,c(2:n.coef)]), data.in$y, alpha = 0, 
                       lambda = opt.lambda, data = data.in)
    ridge.est = c(ridge.fit$a0, as.numeric(ridge.fit$beta))
    
    ## LASSO
    lasso.fit = Lasso(x = data.in[, 2:n.coef], y = data.in$y, fix.lambda = FALSE,
                      cv.method = "cv1se",
                      intercept = TRUE)
    lasso.est = c(lasso.fit$beta0, lasso.fit$beta)
    # lasso.y.est = sum(mypredict(lasso.fit, newx = x.obs[,2:n.coef]))
    #lasso.y.est = mean(x.obs%*%as.matrix(lasso.est))
    
  } else{
    # ghat.y.est = lasso.y.est = 0
    ridge.est = lasso.est = rep(0, 4)
  }  
  
  ## OLS
  lm.fit = lm(y~1+x2+x3+x4, data = data.in)
  lm.est = lm.fit$coefficients
  # lm.y.est = mean(lm.fit$fitted.values)
  
  ## WOLS
  data.w = data.in
  data.w$w = pmin(1/(lm.fit$fitted.values)^2, 10^5)
  
  lm.w.fit = lm(y~1+x2+x3+x4, data = data.w, weights = w)
  lm.w.est = lm.w.fit$coefficients
  # lm.w.y.est = mean(lm.w.fit$fitted.values)
  
  ## Quantile regression
  qr.fit <- rq(y~1+x2+x3+x4, data = data.in)
  qr.est = qr.fit$coefficients
  
  ## MLE
  if (MLE.in){
    x.mat = x.obs
    
    solve.function = function(temp.est){
      xtbeta = x.mat%*%temp.est
      obj.1 = sapply(1:n.obs, function(ind.obs){
        -x.mat[ind.obs,]/xtbeta[ind.obs]
      })
      obj.2 = sapply(1:n.obs, function(ind.obs){
        (y.obs[ind.obs] - xtbeta[ind.obs])/
          ((xtbeta[ind.obs])^3)*y.obs[ind.obs]*x.mat[ind.obs,]/a.in
      })
      obj.12 = apply(obj.1, 1, sum) + apply(obj.2, 1, sum)
      return(obj.12)
    }
    
    solve.I.function = function(temp.est){
      xtbeta = x.mat%*%temp.est
      obj.1 = sapply(1:n.obs, function(ind.obs){
        -as.matrix(x.mat[ind.obs,])%*%t(as.matrix(x.mat[ind.obs,]))/((xtbeta[ind.obs])^2)
      })
      obj.2 = sapply(1:n.obs, function(ind.obs){
        (-3*y.obs[ind.obs] +2*xtbeta[ind.obs])/
          ((xtbeta[ind.obs])^4)*y.obs[ind.obs]/a.in*
          as.matrix(x.mat[ind.obs,])%*%t(as.matrix(x.mat[ind.obs,]))
      })
      obj.12 = apply(obj.1, 1, sum) + apply(obj.2, 1, sum)
      obj.12.mat = matrix(obj.12, nrow=n.coef,ncol=n.coef,byrow = TRUE)
      return(obj.12.mat)
    }
    
    tol = 10^(-5)
    newton.ind = TRUE
    newton.count = 1
    est.init = rep(0.1, n.coef)
    while(newton.ind){
      est.up = est.init - limSolve::Solve(solve.I.function(est.init))%*%
        as.matrix(solve.function(est.init))
      if (max(abs(est.up-est.init))<tol) newton.ind = FALSE
      if (newton.count>100) newton.ind = FALSE
      est.init = est.up
      newton.count = newton.count + 1
    }
    
    mse.est = est.up
    # mse.est = multiroot(f = solve.function, start = c(0.1, 0.1))$root
  } else{
    mse.est = rep(0, n.coef)
  }
  
  return.list = c(lm.w.est, lm.est, mse.est, lasso.est, ridge.est,qr.est
  )
  # print(return.list)
  
  return(return.list)
  
}

###### case adaptive functions

adaptive.bin.data.func = function(rate.pbo.in, rate.trt.in, theta.cutoff.in, 
                                  n1.in, n2a.in, n2b.in){
  
  rand.1.pbo = rbinom(n1.in, 1, rate.pbo.in)
  rand.1.trt = rbinom(n1.in, 1, rate.trt.in)
  rand.1.delta = mean(rand.1.trt) - mean(rand.1.pbo)
  
  adap.ind = (rand.1.delta > theta.cutoff.in)
  
  if (adap.ind){
    rand.2.pbo = rbinom(n2a.in, 1, rate.pbo.in)
    rand.2.trt = rbinom(n2a.in, 1, rate.trt.in)
  } else {
    rand.2.pbo = rbinom(n2b.in, 1, rate.pbo.in)
    rand.2.trt = rbinom(n2b.in, 1, rate.trt.in)
  }
  
  rand.2.delta = mean(rand.2.trt) - mean(rand.2.pbo)
  
  naive.1.out = rand.1.delta*0.2 + 0.8*rand.2.delta
  naive.2.out = rand.1.delta*0.5 + 0.5*rand.2.delta
  naive.3.out = rand.1.delta*0.8 + 0.2*rand.2.delta
  
  return.stats = c(rand.1.delta,
                   rand.2.delta)
  new.list = c(return.stats,
               naive.1.out,
               naive.2.out,
               naive.3.out,
                mean(rand.1.pbo),
                mean(rand.1.trt),
                mean(rand.2.pbo),
                mean(rand.2.trt),
               length(rand.2.pbo)
                  )

  return(new.list)
}

adaptive.bin.three.data.func = function(rate.pbo.in, rate.trt.in, theta.cutoff.in, 
                                  n1.in, n2a.in, n2b.in){
  
  rand.1.pbo = rbinom(n1.in, 1, rate.pbo.in)
  rand.1.trt = rbinom(n1.in, 1, rate.trt.in)
  rand.1.delta = mean(rand.1.trt) - mean(rand.1.pbo)
  
  adap.ind = (rand.1.delta > theta.cutoff.in)
  
  if (adap.ind){
    rand.2.pbo = rbinom(n2a.in, 1, rate.pbo.in)
    rand.2.trt = rbinom(n2a.in, 1, rate.trt.in)
  } else {
    rand.2.pbo = rbinom(n2b.in, 1, rate.pbo.in)
    rand.2.trt = rbinom(n2b.in, 1, rate.trt.in)
  }
  
  rand.2.cum.delta = mean(c(rand.1.trt, rand.2.trt)) - mean(c(rand.1.pbo, rand.2.pbo))
  adap.2.ind = (rand.2.cum.delta > theta.cutoff.in)
  
  if (adap.2.ind){
    rand.3.pbo = rbinom(n2a.in, 1, rate.pbo.in)
    rand.3.trt = rbinom(n2a.in, 1, rate.trt.in)
  } else {
    rand.3.pbo = rbinom(n2b.in, 1, rate.pbo.in)
    rand.3.trt = rbinom(n2b.in, 1, rate.trt.in)
  }
  
  rand.2.delta = mean(rand.2.trt) - mean(rand.2.pbo)
  rand.3.delta = mean(rand.3.trt) - mean(rand.3.pbo)
  
  naive.1.out = rand.1.delta*0.1 + 0.45*rand.2.delta + 0.45*rand.3.delta
  naive.2.out = rand.1.delta*0.3 + rand.2.delta*0.35 + rand.3.delta*0.35
  naive.3.out = rand.1.delta*0.5 + rand.2.delta*0.25 + rand.3.delta*0.25
  
  return.stats = c(rand.1.delta,
                   rand.2.delta,
                   rand.3.delta)
  new.list = c(return.stats,
               naive.1.out,
               naive.2.out,
               naive.3.out,
               mean(rand.1.pbo),
               mean(rand.2.pbo),
               mean(rand.3.pbo),
               length(rand.2.pbo)
  )
  
  return(new.list)
}

###### other examples


###### DNN functions

DNN.fit = function(data.train.scale.in, data.train.label.in, 
                          drop.rate.in, active.name.in, n.node.in, 
                          n.layer.in, max.epoch.in, batch_size_in, validation.prop.in,
                   n.endpoint.in){
  #k_clear_session()
  build_model <- function(drop.rate.in) {
    model <- NULL
    
    model.text.1 = paste0("model <- keras_model_sequential() %>% layer_dense(units = n.node.in, activation =",
                          shQuote(active.name.in),
                          ",input_shape = dim(data.train.scale.in)[2]) %>% layer_dropout(rate=", drop.rate.in, ")%>%")
    
    model.text.2 = paste0(rep(paste0(" layer_dense(units = n.node.in, activation = ",
                                     shQuote(active.name.in),
                                     ") %>% layer_dropout(rate=", drop.rate.in, ")%>%"),
                              (n.layer.in-1)), collapse ="")
    
    ### model.text.3
    model.text.3 = paste0("layer_dense(units = ",n.endpoint.in,
                          ")")
    # model.text.3 = paste0("layer_dense(units = 1)")
    
    eval(parse(text=paste0(model.text.1, model.text.2, model.text.3)))
    
    model %>% compile(
      # loss = MLAE,
      loss = "mse", 
      # loss = "binary_crossentropy",
      optimizer = optimizer_rmsprop(),
      metrics = list("mse")
      # metrics = c('accuracy')
    )
    
    model
  }
  
  out.model <- build_model(drop.rate.in)
  out.model %>% summary()
  
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 1000 == 0) cat("\n")
      cat(".")
    }
  )  
  
  history <- out.model %>% fit(
    data.train.scale.in,
    data.train.label.in,
    epochs = max.epoch.in,
    validation_split = validation.prop.in,
    verbose = 0,
    callbacks = list(print_dot_callback),
    batch_size = batch_size_in
  )
  return(list("model" = out.model, "history" = history))
}

pred.DNN.normal = function(DNN.final.weights.in, n.layer.in, data.train.in,
                           col_means_train.in, col_stddevs_train.in){
  
  w1.scale = DNN.final.weights.in[[1]]
  b1.scale = as.matrix(DNN.final.weights.in[[2]])
  w1 = t(w1.scale/matrix(rep(col_stddevs_train.in, dim(w1.scale)[2]),
                         nrow = dim(w1.scale)[1], ncol = dim(w1.scale)[2]))
  b1 = b1.scale - t(w1.scale)%*%as.matrix(col_means_train.in/col_stddevs_train.in)
  
  for (wb.itt in 2:(n.layer.in+1)){
    w.text = paste0("w", wb.itt, "=t(DNN.final.weights.in[[", wb.itt*2-1, "]])")
    b.text = paste0("b", wb.itt, "= as.matrix(DNN.final.weights.in[[", wb.itt*2, "]])")
    
    eval(parse(text=w.text))
    eval(parse(text=b.text))
  }
  
  eval_f_whole_text1 = paste0(
    "eval_f <- function( x ) {x.mat = as.matrix(as.numeric(x), nrow = length(x), ncol = 1);
    w1x = (w1)%*%x.mat + b1;sw1x = as.matrix(c(relu(w1x)))")
  
  eval_f_whole_text2 = NULL
  for (wb.itt in 2:(n.layer.in)){
    wx.text = paste0("w", wb.itt, "x = (w", wb.itt, ")%*%sw", wb.itt-1,
                     "x + b", wb.itt)
    swx.text = paste0("sw", wb.itt, "x = as.matrix(c(relu(w", wb.itt, "x)))")
    eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  }
  
  wb.itt.final = n.layer.in + 1
  wx.text = paste0("w", wb.itt.final, "x = (w", wb.itt.final, ")%*%sw", wb.itt.final-1,
                   "x + b", wb.itt.final)
  swx.text = paste0("sw",n.layer.in+1,"x =(w", wb.itt.final, "x)")
  eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  
  eval_f_whole_text3 = paste0(";return(sw", n.layer.in+1, "x)}")
  
  eval_f_whole_text = paste(eval_f_whole_text1, eval_f_whole_text2,
                            eval_f_whole_text3)
  
  eval(parse(text=eval_f_whole_text))
  
  final.pred = sapply(1:(dim(data.train.in)[1]),
                      function(y){eval_f(x = as.vector(data.train.in[y,]))})
  return(final.pred)
} 

DNN.sigmoid.fit = function(data.train.scale.in, data.train.label.in, 
                   drop.rate.in, active.name.in, n.node.in, 
                   n.layer.in, max.epoch.in, batch_size_in, validation.prop.in,
                   n.endpoint.in){
  #k_clear_session()
  build_model <- function(drop.rate.in) {
    model <- NULL
    
    model.text.1 = paste0("model <- keras_model_sequential() %>% layer_dense(units = n.node.in, activation =",
                          shQuote(active.name.in),
                          ",input_shape = dim(data.train.scale.in)[2]) %>% layer_dropout(rate=", drop.rate.in, ")%>%")
    
    model.text.2 = paste0(rep(paste0(" layer_dense(units = n.node.in, activation = ",
                                     shQuote(active.name.in),
                                     ") %>% layer_dropout(rate=", drop.rate.in, ")%>%"),
                              (n.layer.in-1)), collapse ="")
    
    ### model.text.3
    model.text.3 = paste0("layer_dense(units = ",n.endpoint.in, 
                          ", activation = ", shQuote("sigmoid"), ")")
    # model.text.3 = paste0("layer_dense(units = 1)")
    
    eval(parse(text=paste0(model.text.1, model.text.2, model.text.3)))
    
    model %>% compile(
      # loss = MLAE,
      loss = "mse", 
      # loss = "binary_crossentropy",
      optimizer = optimizer_rmsprop(),
      metrics = list("mse")
      # metrics = c('accuracy')
    )
    
    model
  }
  
  out.model <- build_model(drop.rate.in)
  out.model %>% summary()
  
  print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 1000 == 0) cat("\n")
      cat(".")
    }
  )  
  
  history <- out.model %>% fit(
    data.train.scale.in,
    data.train.label.in,
    epochs = max.epoch.in,
    validation_split = validation.prop.in,
    verbose = 0,
    callbacks = list(print_dot_callback),
    batch_size = batch_size_in
  )
  return(list("model" = out.model, "history" = history))
}

pred.DNN.sigmoid = function(DNN.final.weights.in, n.layer.in, data.train.in,
                           col_means_train.in, col_stddevs_train.in){
  
  w1.scale = DNN.final.weights.in[[1]]
  b1.scale = as.matrix(DNN.final.weights.in[[2]])
  w1 = t(w1.scale/matrix(rep(col_stddevs_train.in, dim(w1.scale)[2]),
                         nrow = dim(w1.scale)[1], ncol = dim(w1.scale)[2]))
  b1 = b1.scale - t(w1.scale)%*%as.matrix(col_means_train.in/col_stddevs_train.in)
  
  for (wb.itt in 2:(n.layer.in+1)){
    w.text = paste0("w", wb.itt, "=t(DNN.final.weights.in[[", wb.itt*2-1, "]])")
    b.text = paste0("b", wb.itt, "= as.matrix(DNN.final.weights.in[[", wb.itt*2, "]])")
    
    eval(parse(text=w.text))
    eval(parse(text=b.text))
  }
  
  eval_f_whole_text1 = paste0(
    "eval_f <- function( x ) {x.mat = as.matrix(as.numeric(x), nrow = length(x), ncol = 1);
    w1x = (w1)%*%x.mat + b1;sw1x = as.matrix(c(relu(w1x)))")
  
  eval_f_whole_text2 = NULL
  for (wb.itt in 2:(n.layer.in)){
    wx.text = paste0("w", wb.itt, "x = (w", wb.itt, ")%*%sw", wb.itt-1,
                     "x + b", wb.itt)
    swx.text = paste0("sw", wb.itt, "x = as.matrix(c(relu(w", wb.itt, "x)))")
    eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  }
  
  wb.itt.final = n.layer.in + 1
  wx.text = paste0("w", wb.itt.final, "x = (w", wb.itt.final, ")%*%sw", wb.itt.final-1,
                   "x + b", wb.itt.final)
  swx.text = paste0("sw",n.layer.in+1,"x =(w", wb.itt.final, "x)")
  eval_f_whole_text2 = paste(eval_f_whole_text2, wx.text, swx.text, sep = ";")
  
  eval_f_whole_text3 = paste0(";return(sigmoid(sw", n.layer.in+1, "x))}")
  
  eval_f_whole_text = paste(eval_f_whole_text1, eval_f_whole_text2,
                            eval_f_whole_text3)
  
  eval(parse(text=eval_f_whole_text))
  
  final.pred = sapply(1:(dim(data.train.in)[1]),
                      function(y){eval_f(x = as.vector(data.train.in[y,]))})
  return(final.pred)
} 

relu = function(x){
  return(pmax(0, x))
}

sigmoid = function(x){
  return(1/(1+exp(-x)))
}



















