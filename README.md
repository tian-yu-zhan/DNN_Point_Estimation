# DNN_point_estimation

This is a help file for the R code accompanying a paper with the title “Deep Neural Networks Guided Ensemble Learning for Point Estimation”.
The R code of reproducing simulation results is saved at the folder “sim_1” for Table 1, “sim_2_qr” for Table 2, “sim_3” for Table 3 and Figure 2. With results saved in each folder, one can source “DNN_latex_table.r” file to reproduce those tables and the figure. 
•       Training DNN:
Within a specific folder, source “XX_training.r” to train DNN with its scaling parameters. Note that those files are already saved in each example folder.
•       Generating results for validation:
Within a specific folder, source “XX_validation.r” file to obtain validation results, and then use “DNN_latex_table.r” to generate corresponding table / Figure. 
