library(readxl)
library(ggplot2)
library(lubridate)
library(ggthemes)
library(dplyr)
library(see)
library(ggtext)
library(colorspace)


# import and set up data
results_df = read_excel(path="test_img_176_176_NNLS_results_bottom.xlsx")
n_comps = max(results_df$n_compartments)
eval_df = data.frame(compartment=c(1:n_comps), D_mean=0, f=0)

## subset data frame to calculate mean per compartment and parameter
for (comp in 1:n_comps) {
  results_comp = subset(results_df, compartment == comp, select = c("D", "f"))
  eval_df[comp,] = c(comp, mean(results_comp$D), mean(results_comp$f))
}
print(eval_df)