Warmup
================

``` r
knitr::opts_chunk$set(eval = TRUE, echo = TRUE, dev="png")
library(readr)
```

    ## Warning: package 'readr' was built under R version 4.2.1

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.2.3

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.2.2

    ## Loading required package: lattice

``` r
library(ggplot2)
library(dplyr)
```

    ## Warning: package 'dplyr' was built under R version 4.2.3

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
# Load the wine quality data from a CSV file
wine_data <- read.csv("winequality-red.csv", sep= ";")  

# For reproducibility
set.seed(123)

# Split the data into a training set and a testing set
train_index <- createDataPartition(wine_data$quality, p = 0.7, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]
```

# Linear regression

``` r
# Build linear regression 
model <- lm(quality ~ ., data = train_data)

# Evaluate the performance of the model
prediction_train <- predict(model, newdata = train_data)
rmse_train <- sqrt(mean((train_data$quality - prediction_train)^2))


prediction_test <- predict(model, newdata = test_data)
rmse_test <- sqrt(mean((test_data$quality - prediction_test)^2))

print(paste("Prediction Quality for train set RMSE =", rmse_train))
```

    ## [1] "Prediction Quality for train set RMSE = 0.640688208979915"

``` r
print(paste("Prediction Quality for test set RMSE =", rmse_test))
```

    ## [1] "Prediction Quality for test set RMSE = 0.65989109917522"

I used linear regression to predict the quality of red wine. After
fitting the model, I tried to predict and compare results for both train
and test sets to avoid the over fitting. The following plot shows the
actual vs predicted Wine Quality for test set.

``` r
# Plot the Actual and Predicted Wine Quality for test dataset
ggplot(data = test_data, aes(x = quality, y = prediction_test)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Actual Quality", y = "Predicted Quality") +
  ggtitle(paste("Actual vs. Predicted Wine Quality (RMSE =", round(rmse_test, 2), ")"))
```

    ## `geom_smooth()` using formula = 'y ~ x'

![](Warmup_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

## 10-fold cross-validation

Here, I used 10-fold CV for the problem above to compare the results
which are close to each other.

``` r
# Perform 10-fold cross-validation
num_folds <- 10
cv_results <- train(quality ~ ., data = train_data, method = "lm", trControl = trainControl(method = "cv", number = num_folds))
cv_rmse <- cv_results$results$RMSE

prediction_test_cv <- predict(cv_results, newdata = test_data)
rmse_test_cv <- sqrt(mean((test_data$quality - prediction_test_cv)^2))

print(paste("Model Performance using 10-fold cross-validation RMSE =", cv_rmse))
```

    ## [1] "Model Performance using 10-fold cross-validation RMSE = 0.64769086002939"

``` r
print(paste("Prediction Quality for test set using 10-fold cross-validation RMSE =", rmse_test_cv))
```

    ## [1] "Prediction Quality for test set using 10-fold cross-validation RMSE = 0.65989109917522"

# Classification

For classification problem, I am using a threshold to classify the wine
quality as “high” and “low” which would turn the problem to binary
classification. Knn would then be used to train the model and accuracy
of the model be evaluated by confusion matrix for both training and
testing sets.

``` r
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
```

    ## [1] "confusion matrix for train data"

``` r
print(confusion_matrix_train)
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     High Quality Low Quality
    ##   High Quality          482         117
    ##   Low Quality           100         421
    ##                                          
    ##                Accuracy : 0.8062         
    ##                  95% CI : (0.7819, 0.829)
    ##     No Information Rate : 0.5196         
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.6114         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.2774         
    ##                                          
    ##             Sensitivity : 0.8282         
    ##             Specificity : 0.7825         
    ##          Pos Pred Value : 0.8047         
    ##          Neg Pred Value : 0.8081         
    ##              Prevalence : 0.5196         
    ##          Detection Rate : 0.4304         
    ##    Detection Prevalence : 0.5348         
    ##       Balanced Accuracy : 0.8054         
    ##                                          
    ##        'Positive' Class : High Quality   
    ## 

``` r
# Evaluate the classification model for test set
predictions_test <- predict(c_model, newdata = test_data)
confusion_matrix_test <- confusionMatrix(as.factor(test_data$class), as.factor(predictions_test))
print(paste("confusion matrix for test data"))
```

    ## [1] "confusion matrix for test data"

``` r
print(confusion_matrix_test)
```

    ## Confusion Matrix and Statistics
    ## 
    ##               Reference
    ## Prediction     High Quality Low Quality
    ##   High Quality          197          59
    ##   Low Quality            65         158
    ##                                           
    ##                Accuracy : 0.7411          
    ##                  95% CI : (0.6994, 0.7798)
    ##     No Information Rate : 0.547           
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.4789          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.6534          
    ##                                           
    ##             Sensitivity : 0.7519          
    ##             Specificity : 0.7281          
    ##          Pos Pred Value : 0.7695          
    ##          Neg Pred Value : 0.7085          
    ##              Prevalence : 0.5470          
    ##          Detection Rate : 0.4113          
    ##    Detection Prevalence : 0.5344          
    ##       Balanced Accuracy : 0.7400          
    ##                                           
    ##        'Positive' Class : High Quality    
    ## 

Based on the results, accuracy of predictions using train and test sets
are slightly different. The predictions using train set has better
accuracy with 0.80 in comparison to the test set with accuracy of 0.74.
Seems that model is over fitted here, however, we can investigate more
about the model by changing the quality threshold (quality_threshold \<-
6) and split size of the data (p = 0.7).
