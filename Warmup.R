library(readr)
library(caret)
library(ggplot2)
library(dplyr)

# Load the wine quality data from a CSV file
wine_data <- read.csv("winequality-red.csv", sep= ";")  

# For reproducibility
set.seed(123)

# Split the data into a training set and a testing set
train_index <- createDataPartition(wine_data$quality, p = 0.7, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]

# Build linear regression 
model <- lm(quality ~ ., data = train_data)

# Evaluate the performance of the model
prediction_train <- predict(model, newdata = train_data)
rmse_train <- sqrt(mean((train_data$quality - prediction_train)^2))
print(paste("Prediction Quality for train set RMSE =", rmse_train))

prediction_test <- predict(model, newdata = test_data)
rmse_test <- sqrt(mean((test_data$quality - prediction_test)^2))
print(paste("Prediction Quality for test set RMSE =", rmse_test))

# Plot the Actual and Predicted Wine Quality for test data set
ggplot(data = test_data, aes(x = quality, y = prediction_test)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Actual Quality", y = "Predicted Quality") +
  ggtitle(paste("Actual vs. Predicted Wine Quality (RMSE =", round(rmse_test, 2), ")"))


# Perform 10-fold cross-validation
num_folds <- 10
cv_results <- train(quality ~ ., data = train_data, method = "lm", trControl = trainControl(method = "cv", number = num_folds))
cv_rmse <- cv_results$results$RMSE
print(paste("Model Performance using 10-fold cross-validation RMSE =", cv_rmse))

prediction_test_cv <- predict(cv_results, newdata = test_data)
rmse_test_cv <- sqrt(mean((test_data$quality - prediction_test_cv)^2))
print(paste("Prediction Quality for test set using 10-fold cross-validation RMSE =", rmse_test_cv))

# Plot the Actual and Predicted Wine Quality for test data set using cross-validation
ggplot(data = test_data, aes(x = quality, y = prediction_test_cv)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Actual Quality", y = "Predicted Quality") +
  ggtitle(paste("Actual vs. Predicted Wine Quality (RMSE =", round(rmse_test_cv, 2), ")"))


# Define a quality threshold to classify as "high" or "low" quality
quality_threshold <- 6  

# Create a binary classification target variable
train_data$class <- ifelse(train_data$quality >= quality_threshold, "High Quality", "Low Quality")
test_data$class <- ifelse(test_data$quality >= quality_threshold, "High Quality", "Low Quality")

# Create a classification model (random forest)
c_model <- train(class ~ ., data = train_data, method = "knn")

# Evaluate the classification model for train set
predictions_train <- predict(c_model, newdata = train_data)
confusion_matrix_train <- confusionMatrix(as.factor(train_data$class), as.factor(predictions_train))
print(paste("confusion matrix for train data"))
print(confusion_matrix_train)


# Evaluate the classification model for test set
predictions_test <- predict(c_model, newdata = test_data)
confusion_matrix_test <- confusionMatrix(as.factor(test_data$class), as.factor(predictions_test))
print(paste("confusion matrix for test data"))
print(confusion_matrix_test)
