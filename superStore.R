

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
#----

#LOADING
#----
store <- read.csv('/Users/olinyoder/Desktop/Fall 2023/BSAN 430/Datasets/ProjectSuperstore.csv')

head(store,1)
dim(store)
t(t(sapply(store, class)))
#----

#CLEANING
#----
#convert Education, Marital_Status, Response, and Complain to factors
columns <- c("Education", "Marital_Status", "Response", "Complain")
store[columns] <- lapply(store[columns], factor)

#convert Dt_Customer to date
store$Dt_Customer <- anydate(store$Dt_Customer)
store$Dt_Customer <- as.Date(store$Dt_Customer)

t(t(sapply(store, class)))

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
store$daysCust <- as.numeric(store$daysCust)
#----

#FINAL CLEANING
#----
#drop id and date customer column
store = subset(store, select = -c(Id,Dt_Customer))

#remove NA values
store2 <- na.omit(store)

#write out new dataset
write.csv(store2,file='/Users/olinyoder/Desktop/chipotleNew.csv',fileEncoding = "UTF-8")
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

#test for multicollinearity in the model
vif(saturatedLogisticModel)
#not an issue

#odds ratio
round(exp(coef(saturatedLogisticModel)),2)

#predict probabilities on the test data
test_predicted_probabilities <- predict(saturatedLogisticModel, newdata = test.data, type = "response")

#create a ROC curve object for the test data
roc_obj_test <- roc(test.data$Response, test_predicted_probabilities)

#find the optimal threshold using the Youden's J statistic (highest sum of sens+spec) for the test data
optimal_threshold_test <- coords(roc_obj_test, "best", best.method = "youden")$threshold
optimal_threshold_test

#calculate sensitivity and specificity at the optimal threshold for the test data
optimal_sensitivity_test <- coords(roc_obj_test, best.method = "youden")$sensitivity
optimal_specificity_test <- coords(roc_obj_test, best.method = "youden")$specificity

#create binary predictions based on the optimal threshold for the test data
test_predicted_classes <- ifelse(test_predicted_probabilities >= optimal_threshold_test, 1, 0)

#calculate accuracy on the test data
test_observed_classes <- test.data$Response
test_accuracy <- mean(test_predicted_classes == test_observed_classes)
test_accuracy
  #77% accurate

#FEATURE SELECTION (LASSO)

#splitting data
x <- model.matrix(Response ~ ., train.data)[, -1]
y <- train.data$Response

#fit the lasso penalized regression model
set.seed(123)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")

#coefficients
coef(logistcModel1)

#odds ratio
round(exp(coef(logistcModel1)),2)

#fit the final model on the training data using the optimal lambda
lambda_min <- cv.lasso$lambda.min
logisticModel <- glmnet(x, y, alpha = 1, family = "binomial", lambda = lambda_min)

#create a ROC curve object
x.test <- model.matrix(Response ~ ., test.data)[, -1]
probabilities <- predict(logisticModel, s = lambda_min, newx = x.test, type = "response")
roc_obj <- roc(test.data$Response, probabilities)

#find the optimal threshold using the Youden's J statistic
optimal_threshold <- coords(roc_obj, "best", best.method = "youden")$threshold
optimal_threshold

#calculate the corresponding sensitivity and specificity
optimal_sensitivity <- coords(roc_obj, "best", best.method = "youden")[["sensitivity"]]
optimal_specificity <- coords(roc_obj, "best", best.method = "youden")[["specificity"]]

#create binary predictions based on the optimal threshold
predicted_classes <- ifelse(probabilities > optimal_threshold, 1, 0)

#calculate accuracy
observed_classes <- test.data$Response
accuracy <- mean(predicted_classes == observed_classes)

#accuracy
accuracy
  #77% accuracy

#SIMPLIFY
set.seed(123)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)

#simplest model within 1 standard deviation of optimal lambda
logistcModel2 <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.1se)

#coefficients
coef(logistcModel2)

#odds ratio
round(exp(coef(logistcModel2)),2)

x.test1 <- model.matrix(Response ~ ., test.data)[, -1]
probabilities1 <- predict(logistcModel2, s = cv.lasso$lambda.1se, newx = x.test, type = "response")
roc_obj1 <- roc(test.data$Response, probabilities1)

optimal_threshold1 <- coords(roc_obj1, "best", best.method = "youden")$threshold
optimal_threshold1

predicted_classes1 <- ifelse(probabilities1 > optimal_threshold1, 1, 0)

observed_classes1 <- test.data$Response
accuracy1 <- mean(predicted_classes1 == observed_classes1)
accuracy1
  #74% accuracy
#----

#RANDOM FOREST
#----
#BASE RF
set.seed(123)
rf <- randomForest(Response ~ ., data=train.data, proximity=TRUE)
#rf

p2 <- predict(rf, test.data)
confusionMatrix(p2, test.data$Response)
  #87% accuracy

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

#get optimal mtry and ntree values
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
  }
}

#train model
final_rf <- randomForest(Response ~ ., data = train.data, mtry = best_mtry, ntree = best_ntree, proximity = TRUE)

#test and get accuracy
predicted_classes <- predict(final_rf, newdata = test.data)
final_rf_accuracy <- mean(predicted_classes == test.data$Response)
final_rf_accuracy
  #87% accuracy

varImpPlot(final_rf)
#----

#cluster the data in Jamovi
