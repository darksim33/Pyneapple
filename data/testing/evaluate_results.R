library(readxl)
library(ggplot2)
library(lubridate)
library(ggthemes)
library(dplyr)
library(see)
library(ggtext)
library(colorspace)
library("xlsx")

# Function to calculate mean d and f values for given result data frame
evaluate_results = function(result_df){
  ## Initialize data frame
  n_comps = max(result_df$n_compartments)
  eval_df = data.frame(compartment=c(1:n_comps), d_mean=0, f_mean=0)
  
  ## Subset data frame to calculate mean per compartment and parameter
  for (comp in 1:n_comps) {
    results_comp = subset(result_df, compartment == comp, select = c("D", "f"))
    eval_df[comp,] = c(comp, median(results_comp$D[results_comp$D!=0]), median(results_comp$f[results_comp$f!=0]))
  }
  return(eval_df)
}


# Main script
## Import Excel files and set up data
file.list = list.files(path = "./P11_test", pattern = "*.xlsx", full.names = TRUE)
results_df = lapply(file.list, read_excel)
eval_results = list()

## Prepare excel sheet names
filename.list = list.files(path = "./P11_test", pattern = "*.xlsx", full.names = FALSE)
filename.list = substring(filename.list, first=18) # cut after file name
filename.list = sub("\\..*", "", filename.list) # and cut off file ending

## Evaluate all analysed methods in form of Excel sheets loaded
for (method in seq_along(results_df)) {
  eval_results[[method]] = evaluate_results(results_df[[method]])

  write.xlsx(x=eval_results[[method]], file="evaluated_results_P11.xlsx", sheet = filename.list[method], append = TRUE )
  
}
