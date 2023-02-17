##### PRE-PROCESSING BEFORE MACHINE LEARNING PART #####
#Installing necessary packages
install.packages("dplyr")
install.packages("tidyverse")
install.packages("vctrs")
install.packages("readr")
install.packages("lubridate")
install.packages("rle")

#Loading necessary libraries
library(dplyr) 
library(tidyverse)
library(vctrs)
library(readr)
library(lubridate)
library(rle)

#Importing the dataset
df <- read.csv("~/Desktop/NHH/MASTER/FIE453/FIE453 - Big Data with Applications in Finance/Final Report/df.csv")

#Removing the columns we do not need
df <- df %>%
  select(-LPERMNO, -indfmt, -consol, -popsrc, -datafmt, -autxr, -apc)

#Remove columns without an identificator and fiscal year information
df <- df %>%
  drop_na(GVKEY) %>%
  drop_na(fyear)

#Absolute values for prcc_c, and prcc_f
df$prcc_c <- abs(df$prcc_c)
df$prcc_f <- abs(df$prcc_f)

#Converting the datadate column to date class
df$datadate <- ymd(df$datadate)

#Making a column for interest coverage ratio. If XINT = 0, then ICR > 1.
#MAking another column for Tobin's Q 
df <- df %>%
  mutate(icr = ifelse(xint==0, 1, ((xint+pi-txt)/xint))) %>%
  mutate(tobinsq = (at+(csho*prcc_f)-ceq)/at)

#Rounding the icr and tobinsq to two digits
df$icr <- round(df$icr, digits = 2)
df$tobinsq <- round(df$tobinsq, digits = 2)

#Converting all values to USD 
USD_CAD <- read.csv("~/Downloads/DP_LIVE_02122022112146910.csv")

#Load all currency data from https://data.oecd.org/conversion/exchange-rates.htm
USD_CAD <- USD_CAD %>%
  select(LOCATION, TIME, Value) %>%
  filter(LOCATION == "CAN")

#Make new column
df <- df %>%
  mutate(exch_val = ifelse(curcd == "USD", 1, 0))

#Correct currency for specific currency and year combination
df$exch_val <- case_when((df$curcd == "CAD" & df$fyear == 1971) ~ 1.009811,
                         (df$curcd == "CAD" & df$fyear == 1972) ~ 0.990663,
                         (df$curcd == "CAD" & df$fyear == 1973) ~ 1.000090,
                         (df$curcd == "CAD" & df$fyear == 1974) ~ 0.978018,
                         (df$curcd == "CAD" & df$fyear == 1975) ~ 1.017159,
                         (df$curcd == "CAD" & df$fyear == 1976) ~ 0.986028,
                         (df$curcd == "CAD" & df$fyear == 1977) ~ 1.063442,
                         (df$curcd == "CAD" & df$fyear == 1978) ~ 1.140659,
                         (df$curcd == "CAD" & df$fyear == 1979) ~ 1.171424,
                         (df$curcd == "CAD" & df$fyear == 1980) ~ 1.169227,
                         (df$curcd == "CAD" & df$fyear == 1981) ~ 1.198903,
                         (df$curcd == "CAD" & df$fyear == 1982) ~ 1.233735,
                         (df$curcd == "CAD" & df$fyear == 1983) ~ 1.232412,
                         (df$curcd == "CAD" & df$fyear == 1984) ~ 1.295066,
                         (df$curcd == "CAD" & df$fyear == 1985) ~ 1.365507,
                         (df$curcd == "CAD" & df$fyear == 1986) ~ 1.389471,
                         (df$curcd == "CAD" & df$fyear == 1987) ~ 1.325983,
                         (df$curcd == "CAD" & df$fyear == 1988) ~ 1.230701,
                         (df$curcd == "CAD" & df$fyear == 1989) ~ 1.183972,
                         (df$curcd == "CAD" & df$fyear == 1990) ~ 1.166774,
                         (df$curcd == "CAD" & df$fyear == 1991) ~ 1.145726,
                         (df$curcd == "CAD" & df$fyear == 1992) ~ 1.208723,
                         (df$curcd == "CAD" & df$fyear == 1993) ~ 1.290088,
                         (df$curcd == "CAD" & df$fyear == 1994) ~ 1.365673,
                         (df$curcd == "CAD" & df$fyear == 1995) ~ 1.372445,
                         (df$curcd == "CAD" & df$fyear == 1996) ~ 1.363522,
                         (df$curcd == "CAD" & df$fyear == 1997) ~ 1.384598,
                         (df$curcd == "CAD" & df$fyear == 1998) ~ 1.483505,
                         (df$curcd == "CAD" & df$fyear == 1999) ~ 1.485705,
                         (df$curcd == "CAD" & df$fyear == 2000) ~ 1.485394,
                         (df$curcd == "CAD" & df$fyear == 2001) ~ 1.548840,
                         (df$curcd == "CAD" & df$fyear == 2002) ~ 1.570343,
                         (df$curcd == "CAD" & df$fyear == 2003) ~ 1.401015,
                         (df$curcd == "CAD" & df$fyear == 2004) ~ 1.301282,
                         (df$curcd == "CAD" & df$fyear == 2005) ~ 1.211405,
                         (df$curcd == "CAD" & df$fyear == 2006) ~ 1.134345,
                         (df$curcd == "CAD" & df$fyear == 2007) ~ 1.074046,
                         (df$curcd == "CAD" & df$fyear == 2008) ~ 1.067087,
                         (df$curcd == "CAD" & df$fyear == 2009) ~ 1.141535,
                         (df$curcd == "CAD" & df$fyear == 2010) ~ 1.030113,
                         (df$curcd == "CAD" & df$fyear == 2011) ~ 0.989258,
                         (df$curcd == "CAD" & df$fyear == 2012) ~ 0.999365,
                         (df$curcd == "CAD" & df$fyear == 2013) ~ 1.030137,
                         (df$curcd == "CAD" & df$fyear == 2014) ~ 1.104747,
                         (df$curcd == "CAD" & df$fyear == 2015) ~ 1.278786,
                         (df$curcd == "CAD" & df$fyear == 2016) ~ 1.325615,
                         (df$curcd == "CAD" & df$fyear == 2017) ~ 1.297936,
                         (df$curcd == "CAD" & df$fyear == 2018) ~ 1.295818,
                         (df$curcd == "CAD" & df$fyear == 2019) ~ 1.326793,
                         (df$curcd == "CAD" & df$fyear == 2020) ~ 1.341153,
                         (df$curcd == "CAD" & df$fyear == 2021) ~ 1.253877,
                         (df$curcd == "USD") ~ 1
)

#Replace values with USD exchange rates
df[7:43] <- df[7:43]*df$exch_val
df[46:48] <- df[46:48]*df$exch_val

#Make a new data frame based on industry 
industry <- df %>%
  select(gsector, tobinsq)

#Replace Na with 00 = Others
industry$gsector <- industry$gsector %>%
  replace_na(00)

#Group by gsector and summarise with the median of tobinsq
industry <- df %>%
  group_by(gsector) %>%
  summarise(median_tobinsq = median(tobinsq, na.rm = TRUE))

#Merge industry data frame with df
df <- merge(df, industry, all =  TRUE)

#Making a binary column for each of the requirements of the definition of a 
#zombie company
#1. Interest Coverage Rate Below 1
#2. Tobin's Q less than median for its sector
#3. For more than 2 consecutive years
df <- df %>%
  mutate(bin_icr = ifelse(icr < 1, 1, 0)) %>%
  mutate(bin_tobinsq = ifelse(tobinsq < median_tobinsq, 1, 0))%>%
  arrange(conm, fyear)

df <- df %>%
  mutate(consec2 = with(rle(df$bin_icr == 1 & df$bin_tobinsq==1), rep(as.integer(values & lengths >= 2), lengths)))


test_helm <- df %>%
  filter(conm == "HELMERICH & PAYNE")%>%
  arrange(fyear)

##### IDENTIFY ZOMBIE FIRMS #####

#Data frame filtering how many of the observations fulfill all three requirements
zombie <- df %>%
  filter(consec2 == 1) 

#How many companies is a zombie firm or have been some time during its life time
total_zombie <- n_distinct(zombie$GVKEY)

#Data frame showing existing zombie companies. Must fulfill the requirement
#of icr < 1 and Tobin's Q less than the median for the industry, two consecutive
#years (2020 & 2021) as a condition for consec2 in 2021 also involves 2020
existing_comp <- zombie %>%
  filter(fyear == 2021)

#How many of the companies are a zombie firm today (existing companies) 
zombie_today <- n_distinct(existing_comp$GVKEY)

#Total companies
total_firms <- n_distinct(df$GVKEY)

#Total companies today
firms_today <- df %>%
  filter(fyear == 2021)

firms_today <- n_distinct(firms_today$GVKEY)

#Portion of zombie firms 
total_portion_zombie <- total_zombie/total_firms

#Portion of zombie firms today
portion_zombie_today <- zombie_today/firms_today

#Data frame only containing companies that are still alive today
running <- df %>%
  mutate(alive = if_else(fyear ==2021, 1, 0)) %>%
  filter(alive == 1)

#Finding total zombie companies across year range given bankruptcies
zombie_total <- df %>%
  select(GVKEY, gsector, loc, fyear, conm, bin_icr, bin_tobinsq, consec2)%>%
  group_by(GVKEY) %>%
  mutate(end = max(fyear))%>%
  mutate(zombie = if_else(consec2 == 1 & fyear == max(fyear), 1, 0))

#Count all zombie companies
sum(zombie_total$zombie)

#Making own data frame with only zombie instances 
zombie_total <- zombie_total%>%
  filter(zombie == 1)

##### DESCRIPTIVE STATISTICS #####
#Loading necessary libraries
library(ggplot2)
library(plotly)
library(magrittr)
library(table1)

#Descriptive statistics about distributions 
zombie_total <- zombie_total %>%
  group_by(GVKEY, fyear)

zombie_test <- df %>%
  select(GVKEY, gsector, loc, fyear, conm, bin_icr, bin_tobinsq, consec2)%>%
  group_by(GVKEY) %>%
  mutate(end = max(fyear))%>%
  mutate(start = min(fyear)) %>%
  mutate(zombie = if_else(consec2 == 1 & fyear == max(fyear), 1, 0))%>%
  filter(fyear==2021)

#Changing the start year for the companies before our data set starts to get the 
#correct date of the company
zombie_test["start"][zombie_test["GVKEY"] == "10565"] <- 1956
zombie_test["start"][zombie_test["GVKEY"] == "9225"] <- 1948
zombie_test["start"][zombie_test["GVKEY"] == "9611"] <- 1962
zombie_test["start"][zombie_test["GVKEY"] == "7116"] <- 1945
zombie_test["start"][zombie_test["GVKEY"] == "11537"] <- 1958
zombie_test["start"][zombie_test["GVKEY"] == "5568"] <- 1859
zombie_test["start"][zombie_test["GVKEY"] == "9538"] <- 1918
zombie_test["start"][zombie_test["GVKEY"] == "7750"] <- 1969
zombie_test["start"][zombie_test["GVKEY"] == "3969"] <- 1959
zombie_test["start"][zombie_test["GVKEY"] == "1327"] <- 1962
zombie_test["start"][zombie_test["GVKEY"] == "11220"] <- 1959

zombie_test <- zombie_test %>%
  mutate(age = end - start)

#Descriptive based on location
#Find continent based on countries
#library(countrycode)
country_codes <- read.csv("~/Desktop/NHH/MASTER/FIE453/FIE453 - Big Data with Applications in Finance/country_codes.csv")

country_codes <- country_codes %>%
  select(name, alpha.3, region, sub.region)%>%
  rename(country_name = name)%>%
  rename(loc = alpha.3)

zombie_test <- merge(zombie_test, country_codes)

fig_reg <- zombie_test %>%
  group_by(sub.region) %>%
  mutate(total = 1) %>%
  summarise(zombie_prop = sum(zombie)/sum(total)) %>%
  plot_ly(x = ~sub.region, y = ~zombie_prop, type="bar") %>%
  layout(title = "Proportion of Zombie Companies Based on Sub-Region",
         xaxis = list(title="Sub-Region"),
         yaxis = list(title="Proportions"))

fig_reg

#Show frequency of observations based on sub region
zombie_test <- zombie_test %>%
  mutate(total =1)

aggregate(zombie_test$total, by=list(Category=zombie_test$sub.region), FUN=sum)


#Descriptive based on sector
#Making a new column with the sector names
zombie_test <- zombie_test %>%
  mutate(gsec_name = "")

zombie_test$gsec_name <- case_when(
  zombie_test$gsector == 0 ~ "other",
  zombie_test$gsector == 10 ~ "energy",
  zombie_test$gsector == 15 ~ "materials",
  zombie_test$gsector == 20 ~ "industrials",
  zombie_test$gsector == 25 ~ "consumer discretionary",
  zombie_test$gsector == 30 ~ "consumer staples",
  zombie_test$gsector == 35 ~ "health care",
  zombie_test$gsector == 40 ~ "financials",
  zombie_test$gsector == 45 ~ "information technology",
  zombie_test$gsector == 50 ~ "communication services",
  zombie_test$gsector == 55 ~ "utilities",
  zombie_test$gsector == 60 ~ "real estate",
)

#Bar plot showing the zombie companies distributed based on sectors 
fig_sec <- zombie_test %>%
  group_by(gsec_name) %>%
  mutate(total = 1) %>%
  summarise(zombie_prop = sum(zombie)/sum(total)) %>%
  plot_ly(x = ~gsec_name, y = ~zombie_prop, type="bar") %>%
  layout(title = "Proportion of Zombie Companies Based on Sectors",
         xaxis = list(title="Sector names"),
         yaxis = list(title="Proportions"))

fig_sec

#Descriptive based on age
zombie_test <- zombie_test %>% 
  mutate(
    # Create categories
    age_group = dplyr::case_when(
      age <= 10            ~ "0-10",
      age > 10 & age <= 20 ~ "11-20",
      age > 20 & age <= 30 ~ "21-30",
      age > 30 & age <= 40 ~ "31-40",
      age > 40 & age <= 50 ~ "41-50",
      age > 50             ~ "50 +"
    ))

fig_age <- zombie_test %>%
  group_by(age_group) %>%
  mutate(total = 1) %>%
  summarise(zombie_prop = sum(zombie)/sum(total)) %>%
  plot_ly(x = ~age_group, y = ~zombie_prop, type="bar") %>%
  layout(title = "Proportion of Zombie Companies Based on Age Group",
         xaxis = list(title="Age Group"),
         yaxis = list(title="Proportions"))

fig_age

#Making new columns for leverage, firm size, cash holdings, tangibility, CAPEX, 
#EBIT, and RoA
desc <- df %>%
  mutate(Leverage = (dltt + dlc)/seq) %>%
  mutate(Firm_size = log(at)) %>%
  mutate(Cash_holdings = che/at) %>%
  mutate(Tangibility = ppent/at) %>%
  mutate(CAPEX = capx) %>%
  mutate(EBIT = ebit) %>%
  group_by(GVKEY) %>%
  filter(fyear==2021)

desc <- merge(desc, zombie_test)

#Only selecting the variables we need for the descriptive statistics 
desc <- desc %>%
  select(GVKEY, fyear, Leverage, Firm_size, Cash_holdings, Tangibility, CAPEX, EBIT, tobinsq, age, sale, zombie) %>%
  rename(TobinsQ = tobinsq) %>%
  rename(Age = age)%>%
  rename(Sale = sale)%>%
  rename(Zombie = zombie)%>%
  group_by(GVKEY)%>%
  mutate(Zombie_name = if_else(Zombie == 1, "Zombie", "Non-Zombie"))

desc <- desc[-which(duplicated(desc$GVKEY)), ]

sum(desc$Zombie==1)
sum(desc$Zombie==0)

desc$Firm_size[is.infinite(desc$Firm_size)] <- NA 
desc$TobinsQ[is.infinite(desc$TobinsQ)] <- NA

table1::label(desc$Leverage) <- "Leverage"
table1::label(desc$Firm_size) <- "Firm Size"
table1::label(desc$Cash_holdings) <- "Cash Holdings"
table1::label(desc$Tangibility) <- "Tangibility"
table1::label(desc$CAPEX) <- "CAPEX"
table1::label(desc$EBIT) <- "EBIT"
table1::label(desc$Sale) <- "Sale"
table1::label(desc$Age) <- "Age"

table1::table1(~Leverage+Firm_size+Cash_holdings+Tangibility+CAPEX+EBIT
               +Sale+Age| Zombie_name, data = desc, overall = FALSE)

#Plot over proportion of zombie firms each year 
zombie_propy <- df %>%
  select(fyear, consec2) %>%
  mutate(total = 1) %>%
  group_by(fyear)

zombie_propy <- zombie_propy %>%
  group_by(fyear) %>%
  summarise_each(funs(sum))

zombie_propy <- zombie_propy %>%
  mutate(prop_zombie = (consec2/total)*100)

plot_zombies_year <- zombie_propy %>%
  ggplot(aes(x = fyear, y=prop_zombie)) +
  geom_line()+
  geom_smooth(method = "lm")+
  xlab("Years")+
  ylab("Percentage zombies")+
  ggtitle("Percentage Zombie Firms Based on Years")+
  theme(plot.title = element_text(hjust = 0.5))

plot_zombies_year


#####CLASSIFICATION OF ZOMBIE FIRMS#####

ml_zombie <- df %>%
  select(conm, at, gsector, sale, xint, dltt, dlc, seq, consec2, GVKEY, fyear)%>%
  arrange(conm, fyear)

ml_zombie <- ml_zombie %>%
  mutate(firm_size = log(at)) %>%
  mutate(leverage = (dltt+dlc)/seq) %>%
  group_by(GVKEY) %>%
  mutate(end = max(fyear)) %>%
  mutate(start = min(fyear))%>%
  mutate(zombie_last_year = lag(consec2))

ml_zombie["start"][ml_zombie["GVKEY"] == "10565"] <- 1956
ml_zombie["start"][ml_zombie["GVKEY"] == "9225"] <- 1948
ml_zombie["start"][ml_zombie["GVKEY"] == "9611"] <- 1962
ml_zombie["start"][ml_zombie["GVKEY"] == "7116"] <- 1945
ml_zombie["start"][ml_zombie["GVKEY"] == "11537"] <- 1958
ml_zombie["start"][ml_zombie["GVKEY"] == "5568"] <- 1859
ml_zombie["start"][ml_zombie["GVKEY"] == "9538"] <- 1918
ml_zombie["start"][ml_zombie["GVKEY"] == "7750"] <- 1969
ml_zombie["start"][ml_zombie["GVKEY"] == "3969"] <- 1959
ml_zombie["start"][ml_zombie["GVKEY"] == "1327"] <- 1962
ml_zombie["start"][ml_zombie["GVKEY"] == "11220"] <- 1959

ml_zombie <- ml_zombie %>%
  mutate(age = end - start)


ml_zombie <- ml_zombie %>%
  mutate(gsec_name = "")

ml_zombie$gsector <- ml_zombie$gsector %>% 
  replace(is.na(.), 0)

ml_zombie$gsec_name <- case_when(
  ml_zombie$gsector == 0 ~ "other",
  ml_zombie$gsector == 10 ~ "energy",
  ml_zombie$gsector == 15 ~ "materials",
  ml_zombie$gsector == 20 ~ "industrials",
  ml_zombie$gsector == 25 ~ "consumer discretionary",
  ml_zombie$gsector == 30 ~ "consumer staples",
  ml_zombie$gsector == 35 ~ "health care",
  ml_zombie$gsector == 40 ~ "financials",
  ml_zombie$gsector == 45 ~ "information technology",
  ml_zombie$gsector == 50 ~ "communication services",
  ml_zombie$gsector == 55 ~ "utilities",
  ml_zombie$gsector == 60 ~ "real estate",
)

ml_zombie <- ml_zombie[-1,]

ml_zombie <- ml_zombie %>%
  ungroup(GVKEY)

ml_zombie <- ml_zombie %>%
  select(sale, firm_size, leverage, age, gsec_name, consec2, zombie_last_year) %>%
  rename(sector = gsec_name) %>%
  rename(zombie = consec2)

#Identify missing values
sum(is.na(ml_zombie))

#Impute mean for missing values 

ml_zombie <- ml_zombie %>%
  mutate(across(c(sale, firm_size, leverage), ~replace_na(., median(., na.rm=TRUE))))

ml_zombie$zombie <- ml_zombie$zombie %>% 
  replace(is.na(.), 0)

ml_zombie$zombie_last_year <- ml_zombie$zombie_last_year %>% 
  replace(is.na(.), 0)

sum(is.na(ml_zombie))

ml_zombie <- ml_zombie %>%
  filter_all(all_vars(!is.infinite(.)))

#Encode categorical variables

ml_zombie$zombie <- ifelse(ml_zombie$zombie == 1, "yes", "no") 

ml_zombie$zombie <- factor(ml_zombie$zombie)
ml_zombie$zombie <- factor(ml_zombie$zombie_last_year)

ml_zombie <- ml_zombie %>%
  mutate(sector_communication_services = ifelse(sector == "communication services", 1, 0)) %>%
  mutate(sector_consumer_discretionary = ifelse(sector == "consumer discretionary", 1, 0)) %>%
  mutate(sector_consumer_staples = ifelse(sector == "consumer staples", 1, 0)) %>%
  mutate(sector_energy = ifelse(sector == "energy", 1, 0)) %>%
  mutate(sector_financials = ifelse(sector == "financials", 1, 0)) %>%
  mutate(sector_health_care = ifelse(sector == "health care", 1, 0)) %>%
  mutate(sector_industrials = ifelse(sector == "industrials", 1, 0)) %>%
  mutate(sector_information_technology = ifelse(sector == "information technology", 1, 0)) %>%
  mutate(sector_materials = ifelse(sector == "materials", 1, 0)) %>%
  mutate(sector_other = ifelse(sector == "other", 1, 0)) %>%
  mutate(sector_real_estate = ifelse(sector == "real estate", 1, 0))

ml_zombie <- ml_zombie %>%
  select(-c(sector, sector_communication_services))
#, age_group, age_0_10))

ml_zombie <- as.data.frame(ml_zombie)

ml_zombie <- ml_zombie %>% 
  relocate(zombie, .after = last_col())

#Removing outliers 

boxplot(ml_zombie[,c('leverage','sale','firm_size')], main = "Boxplot of the Continous Features")

Q1_lev <- quantile(ml_zombie$leverage, .25)
Q3_lev <- quantile(ml_zombie$leverage, .75)
IQR_lev <- IQR(ml_zombie$leverage)

ml_zombie <- subset(ml_zombie, ml_zombie$leverage > (Q1_lev - 1.5*IQR_lev) & ml_zombie$leverage < (Q3_lev + 1.5*IQR_lev))
dim(ml_zombie)

Q1_sale <- quantile(ml_zombie$sale, .25)
Q3_sale <- quantile(ml_zombie$sale, .75)
IQR_sale <- IQR(ml_zombie$sale)

ml_zombie <- subset(ml_zombie, ml_zombie$sale > (Q1_sale - 1.5*IQR_sale) & ml_zombie$sale < (Q3_sale + 1.5*IQR_sale))
dim(ml_zombie)

Q1_fs <- quantile(ml_zombie$firm_size, .25)
Q3_fs <- quantile(ml_zombie$firm_size, .75)
IQR_fs <- IQR(ml_zombie$firm_size)

ml_zombie <- subset(ml_zombie, ml_zombie$firm_size > (Q1_fs - 1.5*IQR_fs) & ml_zombie$firm_size < (Q3_fs + 1.5*IQR_fs))
dim(ml_zombie)

#Q1_ca <- quantile(ml_zombie$current_assets, .25)
#Q3_ca <- quantile(ml_zombie$current_assets, .75)
#IQR_ca <- IQR(ml_zombie$current_assets)

#ml_zombie <- subset(ml_zombie, ml_zombie$current_assets > (Q1_ca - 1.5*IQR_ca) & ml_zombie$current_assets < (Q3_ca + 1.5*IQR_ca))
#dim(ml_zombie)

library(ggcorrplot)
model.matrix(~0+., data=ml_zombie) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag = F, type="lower", lab=TRUE, lab_size=2)

par(mfrow = c(2, 2))  # Set up a 2 x 2 plotting space
hist(ml_zombie$sale, breaks = 50, xlab = "Sales", main = "")
hist(ml_zombie$firm_size, breaks = 50, xlab = "Firm Size", main = "")
hist(ml_zombie$leverage, breaks = 50, xlab = "Leverage", main = "")
hist(ml_zombie$age, breaks = 50, xlab = "Age", main = "")

ml_zombie <- ml_zombie %>%
  select(-zombie_last_year)

#Normalize the features
library(caret)
process <- preProcess(as.data.frame(ml_zombie), method=c("range"))

ml_zombie <- predict(process, as.data.frame(ml_zombie))

par(mfrow = c(2, 2))  # Set up a 2 x 2 plotting space
hist(ml_zombie$sale, breaks = 50, xlab = "Sales", main = "")
hist(ml_zombie$firm_size, breaks = 50, xlab = "Firm Size", main = "")
hist(ml_zombie$leverage, breaks = 50, xlab = "Leverage", main = "")
hist(ml_zombie$age, breaks = 50, xlab = "Age", main = "")

#Split the dataset into training and test set 
library(caTools)
set.seed(42)
split = sample.split(ml_zombie, SplitRatio = 0.8)
training_set = subset(ml_zombie, split == TRUE)
test_set = subset(ml_zombie, split == FALSE)

diff <- training_set %>% group_by(zombie) %>% summarize(n=n()) %>% mutate(freq=n/sum(n)) 

par(mfrow = c(1, 1))  # Set up a 2 x 2 plotting space

p1 <- ggplot(diff, aes(x=zombie, fill = zombie, group = zombie)) + geom_bar(aes(y=freq), stat="identity", position = "dodge")+
  ggtitle("Bar Plot Showing the Imbalance")+
  theme(plot.title = element_text(hjust = 0.5))

sum(is.na(ml_zombie))

install.packages("mlr")
library(mlr)

summarizeColumns(training_set)

par(mfrow = c(2, 2))  # Set up a 2 x 2 plotting space
hist(training_set$sale, breaks = 100, xlab = "Sales")
hist(training_set$firm_size, breaks = 100, xlab = "Firm Size")
hist(training_set$leverage, breaks = 100, xlab = "Leverage")
hist(training_set$age, breaks = 100, xlab = "Age")

summary(training_set)
summary(test_set)

install.packages("ROSE")
library(ROSE)

training_set <- ovun.sample(zombie~., data=training_set, method = "under", N = 14872)$data
table(training_set$zombie)

diff2 <- training_set %>% group_by(zombie) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

require(gridExtra)

p2 <- ggplot(diff2, aes(x=zombie, fill = zombie, group = zombie)) + geom_bar(aes(y=freq), stat="identity", position = "dodge")+
  ggtitle("Bar Plot After the Undersampling")+
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(p1, p2, ncol=2)

p1+p2

#Logistic Regression
logit_m =glm(formula = zombie~.,data =training_set ,family='binomial')
summary(logit_m)
logit_P = predict(logit_m , newdata = test_set,type = 'response')
logit_test <- ifelse(logit_P > 0.5,"1","0") # Probability check

logit_test <- factor(logit_test)

levels(logit_test)
levels(test_set$zombie)

cm <- confusionMatrix(data=logit_test, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

par(mfrow=c(1,1))

importances <- varImp(logit_m, scale= FALSE)
importances

plot(importances, top = 20)

library(pROC)
roc_score=roc(test_set$zombie, logit_P) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression")

print(roc_score)


#Random Forest
library(caret)
library(randomForest)

rf_m =randomForest(formula = zombie~. -sale ,data =training_set, importance=TRUE)
summary(rf_m)
rf_P = predict(rf_m , newdata = test_set, class = "prob")

cm <- confusionMatrix(data=rf_P, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

print(rf_P)

rf_P <- as.numeric(as.factor(rf_P))

roc_score=roc(test_set$zombie, rf_P) #AUC score
plot(roc_score ,main ="ROC curve -- Random Forest")

print(roc_score)

 par(mfrow(c(1,1)))

#Decision Trees
library(caret)
library(tree)
library(rpart)
library(rpart.plot)

tree_m <- rpart(formula = zombie~.,data =training_set)
summary(tree_m)
tree_P = predict(tree_m , newdata = test_set, type="class")

class(tree_P)

cm <- confusionMatrix(data=tree_P, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

plot(tree_m)
text(tree_m)

rpart.plot(tree_m)

tree_P <- as.numeric(as.factor(tree_P))

roc_score=roc(test_set$zombie, tree_P) #AUC score
plot(roc_score ,main ="ROC curve -- Decision Tree")

print(roc_score)

#KNN
library(class)
model_prediction_knn = knn(train = training_set[, -15],
             test = test_set[, -15],
             cl = training_set[, 15],
             k = 5,
             prob = TRUE)

cm <- confusionMatrix(data=model_prediction_knn, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

model_prediction_knn <- as.numeric(as.factor(model_prediction_knn))

roc_score=roc(test_set$zombie, model_prediction_knn) #AUC score
plot(roc_score ,main ="ROC curve -- KNN")

print(roc_score)

#SVM
library(e1071)
svm_m = svm(formula = zombie ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

model_prediction_svm = predict(svm_m, newdata = test_set[-15])

cm <- confusionMatrix(data=model_prediction_svm, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

model_prediction_svm <- as.numeric(as.factor(model_prediction_svm))

roc_score=roc(test_set$zombie, model_prediction_svm) #AUC score
plot(roc_score ,main ="ROC curve -- SVM")

print(roc_score)

#Naive Bayes
nb_m = naiveBayes(x = training_set[-15],
                        y = training_set$zombie)

model_prediction_nb = predict(nb_m, newdata = test_set[-15])

cm <- confusionMatrix(data=model_prediction_nb, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

model_prediction_nb <- as.numeric(as.factor(model_prediction_nb))

roc_score=roc(test_set$zombie, model_prediction_nb) #AUC score
plot(roc_score ,main ="ROC curve -- Naïve Bayes")

print(roc_score)

#Radial SVM
rsvm_m = svm(formula = zombie ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

model_prediction_rsvm = predict(rsvm_m, newdata = test_set[-15])

cm <- confusionMatrix(data=model_prediction_rsvm, reference = test_set$zombie)

print(cm)

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Non-Zombie', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Zombie', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Non-Zombie', cex=1.2, srt=90)
  text(140, 335, 'Zombie', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)

model_prediction_rsvm <- as.numeric(as.factor(model_prediction_rsvm))

roc_score=roc(test_set$zombie, model_prediction_rsvm) #AUC score
plot(roc_score ,main ="ROC curve -- Radial SVM")

print(roc_score)


