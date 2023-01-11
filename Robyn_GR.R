robyn_update()

install.packages("reticulate")
install.packages('vctrs')
install.packages('winch')
install.packages('ggplot2')
install.packages('e1071')
install.packages('caTools')
library(dplyr)
library(caTools)
library(e1071)
library(randomForest)
library(ggplot2)
library(winch)
library(reticulate)
library(vctrs)
library(dplyr)
virtualenv_create("r-reticulate")
py_install("nevergrad", pip = TRUE)
use_virtualenv("r-reticulate", required = TRUE)
install.packages('remotes')

remotes::install_github("facebookexperimental/Robyn/R")
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", options(timeout=500), getOption("repos"))))

#### Step 0: Setup environment
library(Robyn)

## Force multicore when using RStudio
Sys.setenv(R_FUTURE_FORK_ENABLE = "true")
options(future.fork.enable = TRUE)

################################################################
#### Step 1: Load data

## Check simulated dataset or load your own dataset
dat_gr<- read.csv("~/Downloads/GR_Data_new.csv", header = TRUE, sep = ';')
dat_gr <- dat_gr %>% slice(-97)
head(dat)

## Check holidays from Prophet
# 59 countries included. If your country is not included, please manually add it.
# Tipp: any events can be added into this table, school break, events etc.
data("dt_prophet_holidays")
head(dt_prophet_holidays)
#outlier finding
#split <- sample.split(dat$FB_Impressions, SplitRatio = 0.7)
#train <- subset(dat, split == TRUE)
#test <- subset(dat, split == FALSE)
#testLabel <- dat$Conversion.Revenue
#model <- svm(train$FB_Impressions, type = 'one-classification', nu = 0.5, scale = TRUE, kernel = 'radial')
#model
#summary(model)

#outliers <- svm$SV
#svm.predtrain<-predict(model,train$FB_Impressions)
#svm.predtest<-predict(model,test$FB_Impressions)

#confTrain<-table(Predicted=svm.predtrain,Reference=trainLabels)
#confTest<-table(Predicted=svm.predtest,Reference=testLabels)

#confusionMatrix(confTest,positive='TRUE')

# Directory where you want to export results to (will create new folders)
robyn_object <- "~/Desktop"
dat_gr <- na.omit(dat_gr)
#log

dat_gr$YT_Impressions <- as.numeric(dat_gr$YT_Impressions)
dat_gr$YT_Clicks <- as.numeric(dat_gr$YT_Clicks)
dat_gr$YT_Cost <- as.numeric(dat_gr$YT_Cost)
dat_gr$FB_Impressions <- as.numeric(dat_gr$FB_Impressions)
dat_gr$FB_Clicks <- as.numeric(dat_gr$FB_Clicks)
dat_gr$FB_Cost <- as.numeric(dat_gr$FB_Cost)
dat_gr$Conversion_revenue <- as.numeric(dat_gr$Conversion_revenue)
dat_gr$Users <- as.numeric(dat_gr$Users)
dat_gr$Conversions <- as.numeric(dat_gr$Conversions)

typeof(dat$FB_Impressions)
dat$Conversion_revenue <- log(dat$Conversion_revenue)
dat$FB_Impressions <- log(dat$FB_Impressions)
dat$FB_Cost <- log(dat$FB_Cost)
dat$YT_Impressions <- log(dat$YT_Impressions)
dat$YT_Cost <- log(dat$YT_Cost)
dat$YT_Clicks <- log(dat$YT_Clicks)
#remove non-positive rows
dat <- dat %>% filter(YT_Cost >= 0)
dat <- dat %>% filter(FB_Cost >= 0)
dat <- dat %>% filter(Conversion_revenue >= 0)
### DEPRECATED: It must have extension .RDS. The object name can be different than Robyn:
# robyn_object <- "~/Desktop/MyRobyn.RDS"

################################################################
#### Step 2a: For first time user: Model specification in 4 steps

#### 2a-1: First, specify input variables

## -------------------------------- NOTE v3.6.0 CHANGE !!! ---------------------------------- ##
## All sign control are now automatically provided: "positive" for media & organic variables
## and "default" for all others. User can still customise signs if necessary. Documentation
## is available in ?robyn_inputs
## ------------------------------------------------------------------------------------------ ##
##search and impressions
## for paid_media_vars, use clicks, impressions etc. NOT SPENDS (only as a last resort).

InputCollect_GR <- robyn_inputs(
  dt_input = dat_gr,
  dt_holidays = dt_prophet_holidays,
  date_var = "Date", # date format must be "2020-01-01"
  dep_var = "Conversion_revenue", # there should be only one dependent variable
  dep_var_type = "revenue", # "revenue" (ROI) or "conversion" (CPA)
  prophet_vars = c("trend", "season", "holiday"), # "trend","season", "weekday" & "holiday"
  prophet_country = "GR", # input one country. dt_prophet_holidays includes 59 countries by default
  context_vars = c("Users", 'Conversions'),#delete this, "Advertising_channel._type"), # e.g. competitors, discount, unemployment etc
  paid_media_spends = c('FB_Cost','YT_Cost'), # mandatory input
  paid_media_vars = c('FB_Impressions','YT_Clicks'), # mandatory.
  # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
  # impressions, GRP etc. If not applicable, use spend instead.
  # factor_vars = c("Advertising_channel._type"), # force variables in context_vars or organic_vars to be categorical
  window_start = "2021-01-03",
  window_end = "2022-10-30",
  adstock = "weibull_pdf" # geometric, weibull_cdf or weibull_pdf.
)
print(InputCollect_GR)

#### 2a-2: Second, define and add hyperparameters

## -------------------------------- NOTE v3.6.0 CHANGE !!! ---------------------------------- ##
## Default media variable for modelling has changed from paid_media_vars to paid_media_spends.
## hyperparameter names needs to be base on paid_media_spends names. Run:
hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
## to see correct hyperparameter names. Check GitHub homepage for background of change.
## Also calibration_input are required to be spend names.
## ------------------------------------------------------------------------------------------ ##

## Guide to setup & understand hyperparameters

## 1. IMPORTANT: set plot = TRUE to see helper plots of hyperparameter's effect in transformation
plot_adstock(plot = FALSE)
plot_saturation(plot = FALSE)

## 2. Get correct hyperparameter names:
# All variables in paid_media_spends and organic_vars require hyperparameter and will be
# transformed by adstock & saturation.
# Run hyper_names() as above to get correct media hyperparameter names. All names in
# hyperparameters must equal names from hyper_names(), case sensitive.
# Run ?hyper_names to check parameter definition.

## 3. Hyperparameter interpretation & recommendation:

## Geometric adstock: Theta is the only parameter and means fixed decay rate. Assuming TV
# spend on day 1 is 100€ and theta = 0.7, then day 2 has 100*0.7=70€ worth of effect
# carried-over from day 1, day 3 has 70*0.7=49€ from day 2 etc. Rule-of-thumb for common
# media genre: TV c(0.3, 0.8), OOH/Print/Radio c(0.1, 0.4), digital c(0, 0.3)

## Weibull CDF adstock: The Cumulative Distribution Function of Weibull has two parameters
# , shape & scale, and has flexible decay rate, compared to Geometric adstock with fixed
# decay rate. The shape parameter controls the shape of the decay curve. Recommended
# bound is c(0.0001, 2). The larger the shape, the more S-shape. The smaller, the more
# L-shape. Scale controls the inflexion point of the decay curve. We recommend very
# conservative bounce of c(0, 0.1), because scale increases the adstock half-life greatly.

## Weibull PDF adstock: The Probability Density Function of the Weibull also has two
# parameters, shape & scale, and also has flexible decay rate as Weibull CDF. The
# difference is that Weibull PDF offers lagged effect. When shape > 2, the curve peaks
# after x = 0 and has NULL slope at x = 0, enabling lagged effect and sharper increase and
# decrease of adstock, while the scale parameter indicates the limit of the relative
# position of the peak at x axis; when 1 < shape < 2, the curve peaks after x = 0 and has
# infinite positive slope at x = 0, enabling lagged effect and slower increase and decrease
# of adstock, while scale has the same effect as above; when shape = 1, the curve peaks at
# x = 0 and reduces to exponential decay, while scale controls the inflexion point; when
# 0 < shape < 1, the curve peaks at x = 0 and has increasing decay, while scale controls
# the inflexion point. When all possible shapes are relevant, we recommend c(0.0001, 10)
# as bounds for shape; when only strong lagged effect is of interest, we recommend
# c(2.0001, 10) as bound for shape. In all cases, we recommend conservative bound of
# c(0, 0.1) for scale. Due to the great flexibility of Weibull PDF, meaning more freedom
# in hyperparameter spaces for Nevergrad to explore, it also requires larger iterations
# to converge.

## Hill function for saturation: Hill function is a two-parametric function in Robyn with
# alpha and gamma. Alpha controls the shape of the curve between exponential and s-shape.
# Recommended bound is c(0.5, 3). The larger the alpha, the more S-shape. The smaller, the
# more C-shape. Gamma controls the inflexion point. Recommended bounce is c(0.3, 1). The
# larger the gamma, the later the inflection point in the response curve.

## 4. Set individual hyperparameter bounds. They either contain two values e.g. c(0, 0.5),
# or only one value, in which case you'd "fix" that hyperparameter.

# Run hyper_limits() to check maximum upper and lower bounds by range
hyper_limits()
# Example hyperparameters ranges for Geometric adstock
#hyperparameters <- list(
#FB_Cost_alphas = c(0.5, 3),
#FB_Cost_gammas = c(0.3, 1),
#FB_Cost_thetas = c(0, 0.3),
#YT_Cost_alphas = c(0.5, 3),
#YT_Cost_gammas = c(0.3, 1),
#YT_Cost_thetas = c(0.1, 0.4)
#)

#correlations
sd_fb <- sd(dat_gr$FB_Impressions)
sd_yt <- sd(dat_gr$YT_Impressions)
cor(dat_gr$FB_Impressions,dat_gr$FB_Cost)
cor(dat_gr$YT_Impressions,dat_gr$YT_Cost)
cor(dat_gr$YT_Clicks,dat_gr$YT_Cost)

# Example hyperparameters ranges for Weibull CDF adstock
#hyperparameters <- list(
#FB_Cost_alphas = c(0.5, 3),
#FB_Cost_gammas = c(0.3, 1),
#FB_Cost_shapes = c(0.0001, 2),
#FB_Cost_scales = c(0, 0.1),
#YT_Cost_alphas = c(0.5, 3),
#YT_Cost_gammas = c(0.3, 1),
#YT_Cost_shapes = c(0.0001, 2),
#YT_Cost_scales = c(0, 0.1)
#)
# Example hyperparameters ranges for Weibull PDF adstock
hyperparameters <- list(
  FB_Cost_alphas = c(0.5, 7),
  FB_Cost_gammas = c(0.3, 1),
  FB_Cost_scales = c(0, 0.1),
  FB_Cost_shapes = c(0.0001, 15),
  YT_Cost_alphas = c(0.5, 7),
  YT_Cost_gammas = c(0.3, 1),
  YT_Cost_scales = c(0, 0.1),
  YT_Cost_shapes = c(0.0001, 15),
  train_size = c(0.5,0.8)
)
#### 2a-3: Third, add hyperparameters into robyn_inputs()

InputCollect_GR <- robyn_inputs(InputCollect = InputCollect_GR, hyperparameters = hyperparameters)
print(InputCollect_GR)

#### 2a-4: Fourth (optional), model calibration / add experimental input

## Guide for calibration

# 1. Calibration channels need to be paid_media_spends or organic_vars names.
# 2. We strongly recommend to use Weibull PDF adstock for more degree of freedom when
# calibrating Robyn.
# 3. We strongly recommend to use experimental and causal results that are considered
# ground truth to calibrate MMM. Usual experiment types are identity-based (e.g. Facebook
# conversion lift) or geo-based (e.g. Facebook GeoLift). Due to the nature of treatment
# and control groups in an experiment, the result is considered immediate effect. It's
# rather impossible to hold off historical carryover effect in an experiment. Therefore,
# only calibrates the immediate and the future carryover effect. When calibrating with
# causal experiments, use calibration_scope = "immediate".
# 4. It's controversial to use attribution/MTA contribution to calibrate MMM. Attribution
# is considered biased towards lower-funnel channels and strongly impacted by signal
# quality. When calibrating with MTA, use calibration_scope = "immediate".
# 5. Every MMM is different. It's highly contextual if two MMMs are comparable or not.
# In case of using other MMM result to calibrate Robyn, use calibration_scope = "total".
# 6. Currently, Robyn only accepts point-estimate as calibration input. For example, if
# 10k$ spend is tested against a hold-out for channel A, then input the incremental
# return as point-estimate as the example below.
# 7. The point-estimate has to always match the spend in the variable. For example, if
# channel A usually has $100K weekly spend and the experimental holdout is 70%, input
# the point-estimate for the $30K, not the $70K.
# 8. If an experiment contains more than one media variable, input "channe_A+channel_B"
# to indicate combination of channels, case sensitive.
sum(dat_gr$FB_Cost)
sum(dat_gr$YT_Cost)
sum(dat_gr$Conversion_revenue)
?robyn_inputs
calibration_input_GR <- data.frame(
  #channel name must in paid_media_vars
  channel = c('FB_Cost','YT_Cost'),
  # liftStartDate must be within input data range
  liftStartDate = as.Date(c("2021-01-03","2021-01-03")),
  # liftEndDate must be within input data range
  liftEndDate = as.Date(c("2022-10-30","2022-10-30")),
  # Provided value must be tested on same campaign level in model and same metric as dep_var_type
  liftAbs = c(817000, 817000),#the amount of Conversion.Revenue attributed to each channel,80% because it's PL
  # Spend within experiment: should match within a 10% error your spend on date range for each channel from dt_input
  spend = c(144044, 1466483),
  # Confidence: if frequentist experiment, you may use 1 - pvalue
  confidence = c(0.85, 0.85),
  # KPI measured: must match your dep_var
  metric = c("Conversion_revenue","Conversion_revenue"),
  # Either "immediate" or "total". For experimental inputs like Facebook Lift, "immediate" is recommended.
  calibration_scope = c("immediate",'total')
)
InputCollect_GR <- robyn_inputs(InputCollect = InputCollect_GR, calibration_input = calibration_input_GR)
print(InputCollect_GR)
################################################################
#### Step 2b: For known model specification, setup in one single step

## Specify hyperparameters as in 2a-2 and optionally calibration as in 2a-4 and provide them directly in robyn_inputs()

# InputCollect <- robyn_inputs(
#   dt_input = dat
#   ,dt_holidays = dt_prophet_holidays
#   ,date_var = "DATE"
#   ,dep_var = "revenue"
#   ,dep_var_type = "revenue"
#   ,prophet_vars = c("trend", "season", "holiday")
#   ,prophet_country = "DE"
#   ,context_vars = c("competitor_sales_B", "events")
#   ,paid_media_spends = c("tv_S", "ooh_S",	"print_S", "facebook_S", "search_S")
#   ,paid_media_vars = c("tv_S", "ooh_S", 	"print_S", "facebook_I", "search_clicks_P")
#   ,organic_vars = c("newsletter")
#   ,factor_vars = c("events")
#   ,window_start = "2016-11-23"
#   ,window_end = "2018-08-22"
#   ,adstock = "geometric"
#   ,hyperparameters = hyperparameters # as in 2a-2 above
#   ,calibration_input = calibration_input # as in 2a-4 above
# )

#### Check spend exposure fit if available
if (length(InputCollect_GR$exposure_vars) > 0) {
  InputCollect_GR$modNLS$plots$FB_Impressions
  InputCollect_GR$modNLS$plots$YT_Clicks
}

##### Manually save and import InputCollect as JSON file
# robyn_write(InputCollect, dir = "~/Desktop")
# InputCollect <- robyn_inputs(
#   dt_input = dt_simulated_weekly,
#   dt_holidays = dt_prophet_holidays,
#   json_file = "~/Desktop/RobynModel-inputs.json")

################################################################
#### Step 3: Build initial model
?robyn_run
## Run all trials and iterations. Use ?robyn_run to check parameter definition
set.seed(123)
OutputModels_GR <- robyn_run(
  InputCollect = InputCollect_GR, # feed in all model specification
  # cores = NULL, # default to max available
  add_penalty_factor = TRUE, # Untested feature. Use with caution.
  iterations = 3000, # 2000 recommended for the dummy dataset with no calibration
  trials = 10, # 5 recommended for the dummy dataset
  outputs = FALSE, # outputs = FALSE disables direct model output - robyn_outputs()
  lambda_control = lambda.min,
  ts_validation = TRUE
)
print(OutputModels_GR)

## Check MOO (multi-objective optimization) convergence plots
OutputModels_GR$convergence$moo_distrb_plot
OutputModels_GR$convergence$moo_cloud_plot
# check convergence rules ?robyn_converge
?robyn_converge
?robyn_clusters
?robyn_outputs
if (OutputModels_GR$ts_validation) OutputModels$ts_validation_plot

## Calculate Pareto optimality, cluster and export results and plots. See ?robyn_outputs
OutputCollect_GR <- robyn_outputs(
  InputCollect_GR, OutputModels_GR,
  pareto_fronts = 'auto',
  calibration_constraint = 0.05, # range c(0.01, 0.1) & default at 0.1
  csv_out = "pareto", # "pareto", "all", or NULL (for none)
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  #min_candidates = 100, # top pareto models for clustering. default to 100
  plot_pareto = TRUE, # Set to FALSE to deactivate plotting and saving model one-pagers
  plot_folder = robyn_object, # path for plots export
  export = TRUE # this will create files locally
)

## 4 csv files are exported into the folder for further usage. Check schema here:
## https://github.com/facebookexperimental/Robyn/blob/main/demo/schema.R
# pareto_hyperparameters.csv, hyperparameters per Pareto output model
# pareto_aggregated.csv, aggregated decomposition per independent variable of all Pareto output
# pareto_media_transform_matrix.csv, all media transformation vectors
# pareto_alldecomp_matrix.csv, all decomposition vectors of independent variables


################################################################
#### Step 4: Select and save the any model

## Compare all model one-pagers and select one that mostly reflects your business reality
print(OutputCollect_GR)
select_model_GR <- "2_303_3" # Pick one of the models from OutputCollect to proceed

#### Since 3.7.1: JSON export and import (faster and lighter than RDS files)
ExportedModel_GR <- robyn_write(InputCollect_GR, OutputCollect_GR, select_model_GR)
print(ExportedModel_GR)

one_page_GR <- robyn_onepagers(
  InputCollect_GR,
  OutputCollect_GR,
  select_model = select_model_GR,
  quiet = FALSE,
  export = TRUE
)
one_page_GR
one_page_GR$`9_196_6`$patches$plots[[4]]
###### DEPRECATED (<3.7.1) (might work)
# ExportedModelOld <- robyn_save(
#   robyn_object = robyn_object, # model object location and name
#   select_model = select_model, # selected model ID
#   InputCollect = InputCollect,
#   OutputCollect = OutputCollect
# )
# print(ExportedModelOld)
# # plot(ExportedModelOld)

################################################################
#### Step 5: Get budget allocation based on the selected model above

## Budget allocation result requires further validation. Please use this recommendation with caution.
## Don't interpret budget allocation result if selected model above doesn't meet business expectation.

# Check media summary for selected model
print(ExportedModel_GR)
?robyn_allocator
# Run ?robyn_allocator to check parameter definition
# Run the "max_historical_response" scenario: "What's the revenue lift potential with the
# same historical spend level and what is the spend mix?"
AllocatorCollect1_GR <- robyn_allocator(
  InputCollect = InputCollect_GR,
  OutputCollect = OutputCollect_GR,
  select_model = select_model_GR,
  scenario = "max_historical_response",
  channel_constr_low = 0.7, #you don't wanna spend less than 70% of historical spend per channel
  channel_constr_up = c(1.2, 1.5),#max budget of each channel when compared to historical spend
  export = FALSE,
  date_min = "2021-01-17",
  date_max = "2022-10-30"
)
print(AllocatorCollect1_GR)
plot(AllocatorCollect1_GR)

# Run the "max_response_expected_spend" scenario: "What's the maximum response for a given
# total spend based on historical saturation and what is the spend mix?" "optmSpendShareUnit"
# is the optimum spend share.
AllocatorCollect2_GR <- robyn_allocator(
  InputCollect = InputCollect_GR,
  OutputCollect = OutputCollect_GR,
  select_model = select_model_GR,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7),
  channel_constr_up = c(1.2, 1.5),
  expected_spend = 1000000, # Total spend to be simulated
  expected_spend_days = 7, # Duration of expected_spend in days
  export = TRUE
)
print(AllocatorCollect2_GR)
AllocatorCollect2_GR$dt_optimOut
plot(AllocatorCollect2_GR)

## A csv is exported into the folder for further usage. Check schema here:
## https://github.com/facebookexperimental/Robyn/blob/main/demo/schema.R

## QA optimal response
# Pick any media variable: InputCollect$all_media
select_media_GR <- "FB_Cost"
# For paid_media_spends set metric_value as your optimal spend
metric_value_GR <- AllocatorCollect1_GR$dt_optimOut$optmSpendUnit[
  AllocatorCollect1_GR$dt_optimOut$channels == select_media_GR
]; metric_value_GR
# # For paid_media_vars and organic_vars, manually pick a value
# metric_value_GR <- 10000

if (TRUE) {
  optimal_response_allocator <- AllocatorCollect1_GR$dt_optimOut$optmResponseUnit[
    AllocatorCollect1_GR$dt_optimOut$channels == select_media_GR
  ]
  optimal_response_GR <- robyn_response(
    InputCollect = InputCollect_GR,
    OutputCollect = OutputCollect_GR,
    select_model = select_model_GR,
    select_build = 0,
    media_metric = select_media_GR,
    metric_value = metric_value_GR
  )
  plot(optimal_response_GR$plot)
  if (length(optimal_response_allocator) > 0) {
    cat("QA if results from robyn_allocator and robyn_response agree: ")
    cat(round(optimal_response_allocator) == round(optimal_response$response), "( ")
    cat(optimal_response$response, "==", optimal_response_allocator, ")\n")
  }
}

################################################################
#### Step 6: Model refresh based on selected model and saved results "Alpha" [v3.7.1]

## Must run robyn_write() (manually or automatically) to export any model first, before refreshing.
## The robyn_refresh() function is suitable for updating within "reasonable periods".
## Two situations are considered better to rebuild model:
## 1. most data is new. If initial model has 100 weeks and 80 weeks new data is added in refresh,
## it might be better to rebuild the model. Rule of thumb: 50% of data or less can be new.
## 2. new variables are added.

# Provide JSON file with your InputCollect and ExportedModel specifications
# It can be any model, initial or a refresh model
json_file <- "~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json"
RobynRefresh <- robyn_refresh(
  json_file = json_file,
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  refresh_steps = 13,
  refresh_iters = 1000, # 1k is an estimation
  refresh_trials = 1
)

json_file_rf1 <- "~/Desktop/Robyn_202208231837_init/Robyn_202208231841_rf1/RobynModel-1_12_5.json"
RobynRefresh <- robyn_refresh(
  json_file = json_file_rf1,
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  refresh_steps = 7,
  refresh_iters = 1000, # 1k is an estimation
  refresh_trials = 1
)

# InputCollect <- RobynRefresh$listRefresh1$InputCollect
# OutputCollect <- RobynRefresh$listRefresh1$OutputCollect
# select_model <- RobynRefresh$listRefresh1$OutputCollect$selectID

###### DEPRECATED (<3.7.1) (might work)
# # Run ?robyn_refresh to check parameter definition
# Robyn <- robyn_refresh(
#   robyn_object = robyn_object,
#   dt_input = dt_simulated_weekly,
#   dt_holidays = dt_prophet_holidays,
#   refresh_steps = 4,
#   refresh_mode = "manual",
#   refresh_iters = 1000, # 1k is estimation. Use refresh_mode = "manual" to try out.
#   refresh_trials = 1
# )

## Besides plots: there are 4 CSV outputs saved in the folder for further usage
# report_hyperparameters.csv, hyperparameters of all selected model for reporting
# report_aggregated.csv, aggregated decomposition per independent variable
# report_media_transform_matrix.csv, all media transformation vectors
# report_alldecomp_matrix.csv,all decomposition vectors of independent variables

################################################################
#### Step 7: Get budget allocation recommendation based on selected refresh runs

# Run ?robyn_allocator to check parameter definition
AllocatorCollect <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7, 0.7, 0.7, 0.7),
  channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5),
  expected_spend = 2000000, # Total spend to be simulated
  expected_spend_days = 14 # Duration of expected_spend in days
)
print(AllocatorCollect)
# plot(AllocatorCollect)

################################################################
#### Step 8: get marginal returns

## Example of how to get marginal ROI of next 1000$ from the 80k spend level for search channel

# Run ?robyn_response to check parameter definition

## -------------------------------- NOTE v3.6.0 CHANGE !!! ---------------------------------- ##
## The robyn_response() function can now output response for both spends and exposures (imps,
## GRP, newsletter sendings etc.) as well as plotting individual saturation curves. New
## argument names "media_metric" and "metric_value" instead of "paid_media_var" and "spend"
## are now used to accommodate this change. Also the returned output is a list now and
## contains also the plot.
## ------------------------------------------------------------------------------------------ ##

# Get response for 80k from result saved in robyn_object
Spend1 <- 60000
Response1 <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "search_S",
  metric_value = Spend1
)
Response1$response / Spend1 # ROI for search 80k
Response1$plot

#### Or you can call a JSON file directly (a bit slower)
# Response1 <- robyn_response(
#   json_file = json_file,
#   dt_input = dt_simulated_weekly,
#   dt_holidays = dt_prophet_holidays,
#   media_metric = "search_S",
#   metric_value = Spend1
# )

# Get response for +10%
Spend2 <- Spend1 * 1.1
Response2 <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "search_S",
  metric_value = Spend2
)
Response2$response / Spend2 # ROI for search 81k
Response2$plot

# Marginal ROI of next 1000$ from 80k spend level for search
(Response2$response - Response1$response) / (Spend2 - Spend1)

## Example of getting paid media exposure response curves
imps <- 50000000
response_imps <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "facebook_I",
  metric_value = imps
)
response_imps$response / imps * 1000
response_imps$plot

## Example of getting organic media exposure response curves
sendings <- 30000
response_sending <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "newsletter",
  metric_value = sendings
)
response_sending$response / sendings * 1000
response_sending$plot

################################################################
#### Optional: recreate old models and replicate results [v3.7.1]

# From an exported JSON file (which is created automatically when exporting a model)
# we can re-create a previously trained model and outputs. Note: we need to provide
# the main dataset and the holidays dataset, which are NOT stored in the JSON file.
# These JSON files will be automatically created in most cases.

############ WRITE ############
# Manually create JSON file with inputs data only
robyn_write(InputCollect, dir = "~/Desktop")

# Manually create JSON file with inputs and specific model results
robyn_write(InputCollect, OutputCollect, select_model)


############ READ ############
# Recreate `InputCollect` and `OutputCollect` objects
# Pick any exported model (initial or refreshed)
json_file <- "~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json"

# Optional: Manually read and check data stored in file
json_data <- robyn_read(json_file)
print(json_data)

# Re-create InputCollect
InputCollectX <- robyn_inputs(
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  json_file = json_file)

# Re-create OutputCollect
OutputCollectX <- robyn_run(
  InputCollect = InputCollectX,
  json_file = json_file,
  export = FALSE)

# Or re-create both by simply using robyn_recreate()
RobynRecreated <- robyn_recreate(
  json_file = json_file,
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  quiet = FALSE)
InputCollectX <- RobynRecreated$InputCollect
OutputCollectX <- RobynRecreated$OutputCollect

# Re-export model and check summary (will get exported in your current working directory)
myModel <- robyn_write(InputCollectX, OutputCollectX, dir = "~/Desktop")
print(myModel)

# Re-create one-pager
myModelPlot <- robyn_onepagers(InputCollectX, OutputCollectX, export = FALSE)
# myModelPlot$`1_204_5`$patches$plots[[6]]

# Refresh any imported model
RobynRefresh <- robyn_refresh(
  json_file = json_file,
  dt_input = InputCollectX$dt_input,
  dt_holidays = InputCollectX$dt_holidays,
  refresh_steps = 6,
  refresh_mode = "manual",
  refresh_iters = 1000,
  refresh_trials = 1
)

# Recreate response curves
robyn_response(
  InputCollect = InputCollectX,
  OutputCollect = OutputCollectX,
  media_metric = "newsletter",
  metric_value = 50000
)