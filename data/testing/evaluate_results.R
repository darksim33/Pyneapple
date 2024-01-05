library(readxl)
library(ggplot2)
library(lubridate)
library(ggthemes)
library(dplyr)
library(see)
library(ggtext)
library(colorspace)

# Function to calculate mean d and f values for given result data frame
evaluate_results = function(result_df){
  ## Initialize data frame
  n_comps = max(result_df$n_compartments)
  eval_df = data.frame(compartment=c(1:n_comps), d_mean=0, f_mean=0)
  
  ## Subset data frame to calculate mean per compartment and parameter
  for (comp in 1:n_comps) {
    results_comp = subset(result_df, compartment == comp, select = c("D", "f"))
    eval_df[comp,] = c(comp, mean(results_comp$D), mean(results_comp$f))
  }
  return(eval_df)
}


# Main script
## Import Excel files and set up data
file.list = list.files(path = "./PyNeapple_results", pattern = "*.xlsx", full.names = TRUE)
results_df = lapply(file.list, read_excel)
eval_results = list()

## Evaluate all analysed methods in form of Excel sheets loaded
for (method in seq_along(results_df)) {
  eval_results[[method]] = evaluate_results(results_df[[method]])
  
  print(file.list[method])
  print(eval_results[[method]])
  
}
