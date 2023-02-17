#Installing necessary packages
install.packages("dplyr")
install.packages("tidyr")
install.packages("tidyverse")
install.packages("readxl")
install.packages("forecast")
install.packages("xts")
install.packages("zoo")
install.packages("ggplot2")
install.packages("seasonal")
install.packages("TSstudio")

#Loading necessary packages
library(dplyr)
library(tidyr)
library(tidyverse)
library(readxl)
library(forecast)
library(xts)
library(zoo)
library(ggplot2)
library(seasonal)
library(TSstudio)

#Import datafiles. We have weekly data as well as monthly data.
salmon_prices <- read_excel("~/Downloads/salmon_prices2.xls")
salmon_demand <- read_excel("~/Downloads/salmon_demand.xlsx")
salmon_monthly <- read_excel("~/Downloads/Salmon_monthly.xls")

#Retrieves relevant columns from the datasets.
#In salmon prices, we are only interested in the prices in NOK. This is weekly data.
salmon_prices <- salmon_prices%>%
  select(NOK)

#In salmon demand, we are only interested in the weight in tonnes. This represents the exported
#amount. Hence, the demand.
salmon_demand <- salmon_demand%>%
  select(Weight)

#In salmon montlhy, we hare interested in the average montlhy prices. 
salmon_monthly <- salmon_monthly%>%
  select(NOK)

#Makes a tine serie out of the data frame with monthly prices
ts_prices <- ts(salmon_monthly, start = c(2006,1), end = c(2021,11), frequency = 12)

#Plots the time series
autoplot(ts_prices) + 
  ggtitle("Price in NOK from 2006 to 2021") +
  xlab("Time (years)") +
  ylab("Price (NOK per kg)") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

#Splits the ts into test and training set in which the training set contains approximately
#70% of the data, while the testset contains approximately 30% of the observations
split_price <- ts_split(ts.obj = ts_prices, sample.out = 57)


train_price <- split_price$train
test_price <- split_price$test

#Uses the function ets() to use the forecasting method triple exponential smoothing for linear trend and seasonality
ets_train <- ets(train_price)
ets_test <- ets(test_price)

#Makes the forecasting plot trying to predict prices three years ahead
ets_plot<-ets(train_price)

ets_plot

ets_plot%>%
  forecast(h = 93, level = 25) %>%
  autoplot()+
  ylab("Salmon price (in NOK)")+
  autolayer(test_price)+
  theme_bw()+
  ggtitle("Triple exponential smoothing - forecast")+
  theme(plot.title = element_text(hjust = 0.5))

#Measures the accuracy. MAPE is approximately 6,5% on the training set and 9% on the test set
accuracy(ets_train)
accuracy(ets_test)

#Making a seasonality plot over prices to catch seasonality trends
ggseasonplot(ts_prices, ylab = "Salmon price", main = "Seasonal plot over salmon prices", col = rainbow (19))+
  theme_bw()

