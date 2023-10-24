

#LIBRARIES
#----
library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)
library(caTools)
library(e1071)
library(car)
library(anytime)
library(pROC)
library(ResourceSelection)
#----

#LOADING
#----
store <- read.csv('FilePath.csv')

head(store,1)
dim(store)
t(t(sapply(store, class)))
#----

#CLEANING PART 1
#----
#recode necessary factor variables
unique(store$Marital_Status)
unique(store$Education)
unique(store$Response)
unique(store$Complain)

#recode Marital_Status
store$Marital_Status[store$Marital_Status != 'Married'] <- 'Single'

#recode Education
store$Education[store$Education == '2n Cycle'] <- 'Graudate'
store$Education[store$Education == 'Master'] <- 'Graudate'
store$Education[store$Education == 'Graduation'] <- 'Undergraudate'
store$Education[store$Education == 'Basic'] <- 'High School'

#convert Education, Marital_Status, Response, and Complain to factors
columns <- c("Education", "Marital_Status", "Response", "Complain")
store[columns] <- lapply(store[columns], factor)

#convert Dt_Customer to date
store$Dt_Customer <- anydate(store$Dt_Customer)
store$Dt_Customer <- as.Date(store$Dt_Customer)

t(t(sapply(store, class)))

#NULL values
sum(is.na(store))
  #24 na values, inspect them

#which rows have NAs?
na_df <- store[rowSums(is.na(store)) > 0,]
#na_df
  #looks to be just income variables that are NA. Since we have NA values account for ~ 1% of the data, remove NAs rather than imputing
#----

#FEATURE ENGINEERING
#----
#create a new column based on number of days since a person has been a customer
#dataset was published in January 2023
store$daysCust <- difftime("2023-1-01", store$Dt_Customer, units = "days")
head(store$daysCust)
class(store$daysCust)
#convert daysCust from difftime to numeric
store$daysCust <- round(as.numeric(store$daysCust),0)
#----

#EDA
#----
#YEAR_BIRTH
hist(store$Year_Birth)
old<- data.frame(RowNumber = 1:nrow(store), year = store$Year_Birth)
old <- head(old[order(old$year), ], 10)
old
#remove the people born in 1893, 1899, and 1900
store <- store[store$Year_Birth >= 1901, ]
hist(store$Year_Birth)

#EDUCATION
ggplot((store), aes(x=Education)) +
  geom_histogram(stat = "count")
#as expected

#MARITAL_STATUS
ggplot((store), aes(x=Marital_Status)) +
  geom_histogram(stat = "count")
#as expected

#INCOME
boxplot(store$Income)
top_10_income <- data.frame(RowNumber = 1:nrow(store), Income = store$Income)
top_10_income <- head(top_10_income[order(-top_10_income$Income), ], 10)
top_10_income

#remove extreme outlier, row 528
store <- store[-528, ]
dim(store)
boxplot(store$Income)
top_10_income1 <- data.frame(RowNumber = 1:nrow(store), Income = store$Income)
top_10_income1 <- head(top_10_income[order(-top_10_income$Income), ], 10)
top_10_income1
#there are still a couple outliers, but none so drastic as before so we'll keep them for now

#KIDHOME
ggplot((store), aes(x=Kidhome)) +
  geom_histogram(stat = "count")
#no place has more than 2 kids at home

#TEENHOME
ggplot((store), aes(x=Teenhome)) +
  geom_histogram(stat = "count")
#no place has more than 2 teens at home

#RECENCY
hist(store$Recency)
#uniform dist

#MNTWINES
hist(store$MntWines)
hist(log(store$MntWines))
boxplot(store$MntWines)

#MNTFRUITS
hist(store$MntFruits)
hist(log(store$MntFruits))
boxplot(store$MntFruits)

#MNTMEAT
hist(store$MntMeatProducts)
hist(log(store$MntMeatProducts))
boxplot(store$MntMeatProducts)

#MNTFISH
hist(store$MntFishProducts)
hist(log(store$MntFishProducts))
boxplot(store$MntFishProducts)

#MNTSWEET
hist(store$MntSweetProducts)
hist(log(store$MntSweetProducts))
boxplot(store$MntSweetProducts)

#MNTGOLD
hist(store$MntGoldProds)
hist(log(store$MntGoldProds))
boxplot(store$MntGoldProds)

#NUMDEALS
hist(store$NumDealsPurchases)
hist(log(store$NumDealsPurchases))
boxplot(store$NumDealsPurchases)

#NUMWEBPURCHASES
hist(store$NumWebPurchases)
hist(log(store$NumWebPurchases))
boxplot(store$NumWebPurchases)
max(store$NumWebPurchases, na.rm = T)
mean(store$NumWebPurchases, na.rm = T)

#NUMCATALOG
hist(store$NumCatalogPurchases)
hist(log(store$NumCatalogPurchases))
boxplot(store$NumCatalogPurchases)

#NUMSTORE
hist(store$NumStorePurchases)
hist(log(store$NumStorePurchases))
boxplot(store$NumStorePurchases)

#NUMWEBVISITS
hist(store$NumWebVisitsMonth)
hist(log(store$NumWebVisitsMonth))
boxplot(store$NumWebVisitsMonth)

#RESPONSE
ggplot((store), aes(x=Response)) +
  geom_histogram(stat = "count")

#COMPLAIN
ggplot((store), aes(x=Complain)) +
  geom_histogram(stat = "count")

#DAYSCUST
hist(store$daysCust)

#there are a few outliers, but nothing too egregious 
#most amount/number of variables are skewed right which is not surprising (outliers). A transformation could be made, but not necessary
#----

#FINAL CLEANING
#----
#duplicate store without dropping any columns
storeClean <- na.omit(store)

#write out new dataset
write.csv(storeClean,file='FilePath',fileEncoding = "UTF-8")

#new data for modeling
#drop id and date customer column
store = subset(store, select = -c(Id,Dt_Customer))

#remove NA values
store2 <- na.omit(store)
#----

#MODELING

#LOGISTIC REGRESSION
#----
#train-test split & further preprocessing
set.seed(123)

training.samples <- store2$Response %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- store2[training.samples, ]
test.data <- store2[-training.samples, ]

#check for multicollinarity
#GGally::ggpairs(store2[,4:20])
  #no obvious multicollinarity

#SATURATED MODEL
saturatedLogisticModel <- glm(Response ~ ., data = train.data, family = "binomial")
summary(saturatedLogisticModel)

#test for multicollinearity in the model
vif(saturatedLogisticModel)
#not too major of an issue, all well below 5

#deviance residuals
deviance_residuals <- residuals(saturatedLogisticModel, type = "deviance")
hist(deviance_residuals, breaks = 20, main = "Deviance Residuals")

sorted_indices <- order(-abs(deviance_residuals))
sorted_deviance_residuals <- deviance_residuals[sorted_indices]
sorted_data <- data.frame(
  DevianceResidual = sorted_deviance_residuals
)
head(sorted_data, 10)
#could remove point 498

#odds ratio
round(exp(coef(saturatedLogisticModel)),2)

#test model
#predict probabilities on the test data
test_predicted_probabilities <- predict(saturatedLogisticModel, newdata = test.data, type = "response")

#create a ROC curve object for the test data
roc_obj_test <- roc(test.data$Response, test_predicted_probabilities)

#find the optimal threshold - Youden's J statistic (highest sum of sens+spec) for the test data
optimal_threshold_test <- coords(roc_obj_test, "best", best.method = "youden")$threshold
optimal_threshold_test

#sensitivity and specificity
optimal_sensitivity_test <- coords(roc_obj_test, best.method = "youden")$sensitivity
optimal_specificity_test <- coords(roc_obj_test, best.method = "youden")$specificity

#create binary predictions based on the optimal threshold for the test data
test_predicted_classes <- ifelse(test_predicted_probabilities >= optimal_threshold_test, 1, 0)

#calculate accuracy on the test data
test_observed_classes <- test.data$Response
test_accuracy <- mean(test_predicted_classes == test_observed_classes)
test_accuracy
  #~76% accurate

#FEATURE SELECTION (LASSO)

#splitting data
x <- model.matrix(Response ~ ., train.data)[, -1]
y <- train.data$Response
x.test <- model.matrix(Response ~ ., test.data)[, -1]
y.test <- test.data$Response

#fit the lasso penalized regression model
set.seed(123)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")

#fit the final model on the training data using the optimal lambda
lambda_min <- cv.lasso$lambda.min
logisticModel <- glmnet(x, y, alpha = 1, family = "binomial", lambda = lambda_min)
summary(logisticModel)

#multicollinearity
coef_matrix <- as.matrix(coef(logisticModel))
cor_diag <- diag(cor(x))
vif_values <- 1 / (1 - coef_matrix^2)
vif_values
#teenhome VIF > 5

#deviance residuals
dev_predicted_probabilities1 <- predict(logisticModel, s = lambda_min, newx = x.test, type = "response")
head(as.numeric(y.test))
mean(as.numeric(y.test))
#converting to numeric is increasing y by 1, so we need to undo it
y_numeric <- as.numeric(y.test) - 1
deviance_residuals1 <- y_numeric - dev_predicted_probabilities1
hist(deviance_residuals1, breaks = 20, main = "Deviance Residuals")
#no issues

#coefficients
coef(logisticModel)

#odds ratio
round(exp(coef(logisticModel)),2)

#create a ROC curve object
probabilities <- predict(logisticModel, s = lambda_min, newx = x.test, type = "response")
roc_obj <- roc(test.data$Response, probabilities)

#find the optimal threshold - Youden's J statistic (highest sum of sens+spec) for the test data
optimal_threshold <- coords(roc_obj, "best", best.method = "youden")$threshold
optimal_threshold

#sensitivity and specificity
optimal_sensitivity <- coords(roc_obj, "best", best.method = "youden")[["sensitivity"]]
optimal_specificity <- coords(roc_obj, "best", best.method = "youden")[["specificity"]]

#create binary predictions based on the optimal threshold
predicted_classes <- ifelse(probabilities > optimal_threshold, 1, 0)

#calculate accuracy
observed_classes <- test.data$Response
accuracy <- mean(predicted_classes == observed_classes)

#accuracy
accuracy
  #~79% accuracy

#SIMPLIFY
#should take care of issues with multicollinearity and potential overfitting
set.seed(123)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)

#simplest model within 1 standard deviation of optimal lambda
logisticModel2 <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.1se)

#multicollinearity
coef_matrix1 <- as.matrix(coef(logisticModel2))
cor_diag <- diag(cor(x))
vif_values1 <- 1 / (1 - coef_matrix1^2)
vif_values1
#no issues with multicollinearity

#deviance residuals
dev_predicted_probabilities2 <- predict(logisticModel2, s = lambda_min, newx = x.test, type = "response")
#converting to numeric is increasing y by 1, so we need to undo it
y_numeric1 <- as.numeric(y.test) - 1
deviance_residuals2 <- y_numeric1 - dev_predicted_probabilities2
hist(deviance_residuals2, breaks = 20, main = "Deviance Residuals")
#no issues

#coefficients
coef(logisticModel2)

#odds ratio
round(exp(coef(logisticModel2)),2)

probabilities1 <- predict(logisticModel2, s = cv.lasso$lambda.1se, newx = x.test, type = "response")
roc_obj1 <- roc(test.data$Response, probabilities1)

optimal_threshold1 <- coords(roc_obj1, "best", best.method = "youden")$threshold
optimal_threshold1

predicted_classes1 <- ifelse(probabilities1 > optimal_threshold1, 1, 0)

observed_classes1 <- test.data$Response
accuracy1 <- mean(predicted_classes1 == observed_classes1)
accuracy1
  #~75% accuracy
#----

#RANDOM FOREST
#----
#BASE RF
set.seed(123)
rf <- randomForest(Response ~ ., data=train.data, proximity=TRUE)
#rf

p2 <- predict(rf, test.data)
confusionMatrix(p2, test.data$Response)
  #~87% accuracy

varImpPlot(rf)

#OPTIMAL RF
#mtry values
mtry_values <- c(seq(3,7))

#ntree values
ntree_values <- c(seq(100, 1000, by = 100))

#initialize variables
best_mtry <- NULL
best_ntree <- NULL
best_oob_error <- Inf

#optimize mtry and ntree
results_df <- data.frame(mtry = integer(0), ntree = integer(0), oob_error = double(0))

for (mtry in mtry_values) {
  for (ntree in ntree_values) {
    # Train a random forest model with the current mtry and ntree values
    rf_model <- randomForest(Response ~ ., data = train.data, mtry = mtry, ntree = ntree)
    
    # Calculate out-of-bag error
    oob_error <- rf_model$err.rate[nrow(rf_model$err.rate), 1]
    
    # Check if this combination results in a lower OOB error
    if (oob_error < best_oob_error) {
      best_mtry <- mtry
      best_ntree <- ntree
      best_oob_error <- oob_error
    }
    
    # Add results to the data frame
    results_df <- rbind(results_df, data.frame(mtry = mtry, ntree = ntree, oob_error = oob_error))
  }
}

best_mtry
best_ntree

#scatter plot of ntree vs. mtry vs. OOB error rate
ggplot(results_df, aes(x = mtry, y = ntree, color = oob_error)) +
  geom_point() +
  scale_color_gradient(low = "black", high = "white") +
  labs(x = "mtry", y = "ntree", color = "OOB Error") +
  theme_minimal()
#can use the simplest amount of mtry + ntree combo -- simplest combo that is dark on the chart

#train model
final_rf <- randomForest(Response ~ ., data = train.data, mtry = best_mtry, ntree = best_ntree, proximity = TRUE)

#test and get accuracy
predicted_classes <- predict(final_rf, newdata = test.data)
final_rf_accuracy <- mean(predicted_classes == test.data$Response)
final_rf_accuracy
  #~87% accuracy

varImpPlot(final_rf)
#----

#cluster the data in Jamovi
