---
title: "R Machine learning week 4"
output:
  html_document:
    df_print: paged
---

# Data Preprocessing  
```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(ggplot2)
library(dplyr)
```
# Download the Data
```{r}
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(url=UrlTest, destfile = "Test.csv") 
download.file(url=UrlTrain, destfile = "Train.csv")  

dt_training <- read.csv(url(UrlTrain), na.strings = c("#DIV/0!", "NA")) 
dt_testing  <- read.csv(url(UrlTest), na.strings =c("#DIV/0!", "NA")) 
dim(dt_training)
dim(dt_testing)
```
## Cleaning the Data
# Remove all columns that contains NA and remove features that are not in the testing dataset. 
```{r}
dt_training <- dt_training[, colSums(is.na(dt_training)) == 0]
dt_testing <- dt_testing[, colSums(is.na(dt_testing)) == 0]
dim(dt_training)
dim(dt_testing)
```

```{r}
classe <- dt_training$classe
trainRemove <- grepl("^X|timestamp|window", names(dt_training))
dt_training <- dt_training[, !trainRemove]
trainCleaned <- dt_training[, sapply(dt_training, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(dt_testing))
dt_testing <- dt_testing[, !testRemove]
testCleaned <- dt_testing[, sapply(dt_testing, is.numeric)]
dim(trainCleaned)
dim(testCleaned)
```

## Partitioning the Dataset
Then, we can split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps
  
```{r}
set.seed(22519) 
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
training <- trainCleaned[inTrain, ]
testing <- trainCleaned[-inTrain, ]
dim(training)
dim(testing)
```
## Correlation Matrix Visualization
```{r}
corMatrix <- cor(training[, -53])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```
## Data Modeling
# The Random Forest Model
We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. 

```{r}
set.seed(12345)
controlRF <- trainControl(method = "cv", 5)
modFitRF <- train(classe ~ ., data = training, method="rf", trControl = controlRF, ntree = 250)
modFitRF
```
So, the estimated accuracy of the model is 99.17% and the estimated out-of-sample error is 0.58%.

Then, we estimate the performance of the model on the validation data set.  
```{r}
predictRF <- predict(modFitRF, testing)
table(predictRF)
table(testing$class)
confusionMatrixRF <- confusionMatrix(table(testing$classe, predictRF)) 
confusionMatrixRF
```

```{r}
accuracy<-confusionMatrixRF$overall[[1]] 
accuracy
oose <- 1 - as.numeric(confusionMatrix(table(testing$classe, predictRF))$overall[1])
oose
```

# The Decision Tree Model
```{r}
modFitDT <- rpart(classe ~ ., data = training, method="class")
prp(modFitDT)
```
```{r}
predictDT <- predict(modFitDT,testing, type="class") 
confMatDT <- confusionMatrix(table(factor(predictDT), factor(testing$classe)))
confMatDT
```
```{r}
table(predictDT)
table(testing$class)
```


