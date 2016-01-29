# Coursera - Practical Machine Learning Course Project
Larry Riggen  
January 2016  
**Project Overview**

   This project is based on the data collected for the paper (http://groupware.les.inf.puc-rio.br/har). The study described in the
   paper had subjects perform exercises with both proper and (several different) improper forms. While performing the exercises, the
   participants were instrumented with wearable accelerometers.
   
   Machine learning algorithms will be use to predict the correctness of the exercise form based on the accelerometer data.
   


**General Analysis Flow**

   1. Data cleaning - there are 160 variables in the dataset. Many of these variable have all missing values and will be removed. Also remove other variables that don't contribute to the analysis.
  
   2. The training set will be separated into a validation and training set so that models can be compared prior to use on the test set.
   
   3. Several different machine learning techniques will be used to create models on the training set and evaluated on the validation set.
   
   4. The model which is the most accurate on the validation set will be used to provide a prediction on the testing set.


**1. Data Cleaning**

   Many of the columns in the dataset consist of NAs. Any column consisting more than 95% NAs will be removed. Also, remove the following columns X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window as they are not needed for the analysis.



```r
# Perform some housekeeping
setwd("E:/Coursera Data Scientist/Practical Machine Learning/Course Project")
library(caret)
library(rpart)
library(data.table)
library(rattle)
set.seed(1492)
# Read the files
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","./pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","./pml-testing.csv")
originalTrainingFile<-read.csv("./pml-training.csv", na.strings=c("NA","#DIV/0!",""))
originalTestingFile<-read.csv("./pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
# Determine the columns containing > 95% NAs and remove them
countNAs<-sapply(originalTrainingFile, function(x) {sum(is.na(x))})
NAcolumns<-names(countNAs[(countNAs/nrow(originalTrainingFile))>.95])
originalTrainingFileColSubSet <- originalTrainingFile[, !names(originalTrainingFile) %in% NAcolumns]
# Remove other non-contributing columns
library(ISLR)
originalTrainingFileColSubSet<-subset(originalTrainingFileColSubSet,select=-c(X,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))
```

**2. Separate the original training set into a training and validation set **


```r
intTrain<-createDataPartition(y=originalTrainingFileColSubSet$classe, p=0.7, list=FALSE)
training<-originalTrainingFileColSubSet[intTrain, ]
testing<-originalTrainingFileColSubSet[-intTrain, ]
```

**3. Build several models and compare them on the validation set **

Create a decision tree, a random forest, and a generalized boosted model based on the subset of the original training dataset assigned
for training.


```r
modFitDT<-rpart(classe~.,method="class",data=training)
modFitRF<-train(classe~.,method="rf",data=training,verbose=FALSE)
modFitGBM<-train(classe~.,method="gbm",data=training,verbose=FALSE)
```

Run the model against the subset of the original training set assigned to testing. Compare the accuracy of the models using the confusion matrix.



```r
predDT<-predict(modFitDT,testing,type="class")
confusionMatrix(predDT,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1573  246   23   98   35
##          B   31  528   40   18   46
##          C   33  220  896   98  140
##          D   22   81   63  668   72
##          E   15   64    4   82  789
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7568          
##                  95% CI : (0.7457, 0.7678)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6909          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9397  0.46356   0.8733   0.6929   0.7292
## Specificity            0.9045  0.97155   0.8990   0.9516   0.9656
## Pos Pred Value         0.7965  0.79638   0.6460   0.7373   0.8270
## Neg Pred Value         0.9742  0.88300   0.9711   0.9406   0.9406
## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
## Detection Rate         0.2673  0.08972   0.1523   0.1135   0.1341
## Detection Prevalence   0.3356  0.11266   0.2357   0.1540   0.1621
## Balanced Accuracy      0.9221  0.71756   0.8861   0.8223   0.8474
```

```r
predRF<-predict(modFitRF,testing)
confusionMatrix(predRF,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   11    0    0    0
##          B    1 1124    3    0    0
##          C    1    4 1022    9    1
##          D    0    0    1  955    2
##          E    1    0    0    0 1079
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9868   0.9961   0.9907   0.9972
## Specificity            0.9974   0.9992   0.9969   0.9994   0.9998
## Pos Pred Value         0.9935   0.9965   0.9855   0.9969   0.9991
## Neg Pred Value         0.9993   0.9968   0.9992   0.9982   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1910   0.1737   0.1623   0.1833
## Detection Prevalence   0.2858   0.1917   0.1762   0.1628   0.1835
## Balanced Accuracy      0.9978   0.9930   0.9965   0.9950   0.9985
```

```r
predGBM<-predict(modFitGBM,testing)
confusionMatrix(predGBM,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1635   28    0    3    1
##          B   22 1072   30    2    9
##          C   11   39  985   38    9
##          D    6    0    8  918   16
##          E    0    0    3    3 1047
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9613        
##                  95% CI : (0.956, 0.966)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.951         
##  Mcnemar's Test P-Value : 1.144e-08     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9767   0.9412   0.9600   0.9523   0.9677
## Specificity            0.9924   0.9867   0.9800   0.9939   0.9988
## Pos Pred Value         0.9808   0.9445   0.9104   0.9684   0.9943
## Neg Pred Value         0.9908   0.9859   0.9915   0.9907   0.9928
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2778   0.1822   0.1674   0.1560   0.1779
## Detection Prevalence   0.2833   0.1929   0.1839   0.1611   0.1789
## Balanced Accuracy      0.9846   0.9640   0.9700   0.9731   0.9832
```

**4. The random forest model was the most accurate. Use it to perform a prediction on the original testing set**



```r
predRF_Final<-predict(modFitRF,originalTestingFile)
predRF_Final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


**Summary**

The predRF_Final set above was used to answer the questions in the Course Project Prediction Quiz with 100% accuracy.
