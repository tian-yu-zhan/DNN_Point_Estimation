
setwd("~/Dropbox/Research/R_code_DNN_point_estimation/R_code_to_share_final/")
library(xtable)
library(ggplot2)
library(latex2exp)
library(labeling)

###################################################################################################
## Table 1 of Section 5.1

# data.in.temp = read.csv(file = "sim_1/results_sim_1_n_10.csv")
# # data.in.temp = read.csv(file = "sim_1/results_sim_1_n_2.csv")
# data.in = data.in.temp[!is.na(match(data.in.temp$theta, c(0.5, 1, 5, "min"))), ]
# 
# data.out = data.frame(
#                       "k" = sprintf('%.1f',data.in$k),
#                       "theta" = data.in$theta,
#                       # "DNN_bias" = data.in$bias_DNN,
#                       "space" = "",
#                       # "DNN_bias_10_3" = sprintf('%.4f',data.in$bias_DNN/data.in$theta*10^3),
#                       "DNN_sd" = sprintf('%.3f',sqrt(data.in$MSE_DNN)),
#                       "space" = "",
#                       "ratio_rb" = sprintf('%.3f',data.in$MSE_ratio_rb),
#                       "ratio_js" = sprintf('%.3f',data.in$MSE_ratio_js),
#                       "ratio_half" = sprintf('%.3f',data.in$MSE_ratio_half),
#                       "ratio_mle" = sprintf('%.3f',data.in$MSE_ratio_mle),
#                       "ratio_mean" = sprintf('%.3f',data.in$MSE_ratio_mean)
#                       )
# 
# print(xtable(data.out),include.rownames = FALSE)

################################################################################################
## Table 2 of Section 5.2
# # data.in.temp = read.csv(file = "sim_2/results_sim_2_first.csv")
# data.in.temp = read.csv(file = "sim_2_qr/results_sim_2.csv")
# 
# data.in = data.in.temp[c(1:3, 7:9), ]
# 
# data.out = data.frame(
#   "beta1" = sprintf('%.1f',data.in$beta_1),
#   "beta2" = sprintf('%.1f',data.in$beta_2),
#   "beta3" = sprintf('%.1f',data.in$beta_3),
#   "beta4" = sprintf('%.1f',data.in$beta_4),
#   "space" = "",
#   "RE_qr" = sprintf('%.3f',data.in$MSE_qr_total/data.in$MSE_DNN_total),
#   "RE_wlse" = sprintf('%.3f',data.in$MSE_wlse_total/data.in$MSE_DNN_total),
#   "RE_lse" = sprintf('%.3f',data.in$MSE_lse_total/data.in$MSE_DNN_total)
# )
# 
# print(xtable(data.out),include.rownames = FALSE)

################################################################################################
# ## Table 3 of Section 6
data.in.all = read.csv(file = "sim_3/results_sim_3_n1_100_n2a_50_n2b_250_theta_0.16_results.csv")
data.in.all = data.frame(data.in.all)

data.in.all = data.in.all[c(2, 10, 11, 12), ]

data.out = data.frame(
  "rate_p" = data.in.all$rate_p,
  "rate_t" = data.in.all$rate_t,
  "rate_delta" = data.in.all$rate_delta,
  "space" = "",
  "bias_DNN" = sprintf('%.3f',(data.in.all$bias_DNN)),
  "sd_DNN" = sprintf('%.3f',sqrt(data.in.all$MSE_DNN)),
  "space" = "",
  "sd_1" = sprintf('%.3f',(data.in.all$MSE_ratio_1_DNN)),
  "sd_2" = sprintf('%.3f',(data.in.all$MSE_ratio_2_DNN)),
  "sd_3" = sprintf('%.3f',(data.in.all$MSE_ratio_3_DNN))
)

print(xtable(data.out),include.rownames = FALSE)

#########################################################
## plot
data.in.all = read.csv(file = "sim_3/results_sim_3_n1_100_n2a_50_n2b_250_theta_0.16_results.csv")
data.plot = data.in.all[data.in.all$rate_p==0.47&data.in.all$rate_delta>=0.08, ]
data.plot.long = data.frame("trt_diff" = rep(data.plot$rate_delta, 4),
                            "Estimator" = rep(c("DNN", "theta(0.2)", "theta(0.5)", "theta(0.8)"),
                                              each = 4),
                            "power" = c(data.plot$dec_DNN_0.064, data.plot$dec_1_0.064,
                                        data.plot$dec_2_0.068, data.plot$dec_3_0.094)*100
)

png("sim_3/power_plot.png", width = 1800, height = 1200)
ggplot.fit = ggplot(data.plot.long, aes(x = trt_diff, y = power)) +
  geom_point(size = 8) +
  geom_line(size =2, aes(linetype = Estimator)) +  #plot flow
  # #scale_linetype_manual(values=c("dotted", "solid")) +
  # scale_linetype_manual(values=c("solid", "dotted", "dashed", "dotdash")) +
  # scale_shape_manual(values=c(15:18))+
  # scale_alpha_manual(values=c(1, 1, rep(0.7, 5)))+
  # scale_color_manual(values=
  #                      c("black", "#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00", "#CC79A7"))+
  # # scale_y_continuous(breaks = c(0.75, 0.8, 0.85, 0.9), limits = c(0.65, 0.9), sec.axis = sec_axis(~.*0.5-0.65/2, name = "Type I error", breaks = c(0, 0.01, 0.025, 0.04))) +
  scale_y_continuous(breaks = c(40, 50, 60, 70, 80, 90, 100), limits = c(40, 100)) +
  scale_linetype_manual(name = "Estimator", labels =
                          unname(TeX(c("$U\\[\\widetilde{\\theta}(0.5), \\Delta^{(1)} \\]",
                                       "$\\widetilde{\\theta}(0.2)",
                                       "$\\widetilde{\\theta}(0.5)",
                                       "$\\widetilde{\\theta}(0.8)"))),
                        values=c("solid", "dotted", "dashed", "dotdash")) +
  # # scale_y_continuous(sec.axis = sec_axis(~., name = "Type I error")) +
  # scale_x_continuous(breaks = c(5, 20, 35, 50, 65, 80, 95))+
  # labs(title = "") +
  ylab ("Power (%)") + xlab("Treatment effect") +
  theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=45),
        axis.text.x = element_text(colour="black",size=45,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=45,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=45,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=40,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 37, face = "plain"
        ),
        legend.key.width = unit(6, "line"),
        legend.title = element_text(colour="black", size = 42, face = "plain"),
        legend.key.size = unit(2,"line"),
        legend.position="bottom", plot.title = element_text(hjust = 0.5))

print(ggplot.fit)

dev.off()














