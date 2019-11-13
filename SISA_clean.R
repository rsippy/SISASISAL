#file location
setwd("~/SISA/")

#sorry this code might be ugly in places. I'm self-taught

#data are available as csv files and variables will need to be
#converted to numeric, facter, etc. 

load("~/SISA/sisa.RData")

#the caret library houses all of the machine learning machinery
library(caret)

##################################################
# step 1: tidy up your variables for use in caret
##################################################

#outcome variable should by yes/no
#get rid of any variables you aren't using
sisa$hosp<-"no"
sisahosp[sisaHospitalized==1]<-"yes"
sisa$hosp<-factor(sisa$hosp,levels=c("no","yes"))
sisa<-sisa[,-1]

#missing, linear dependent variables dropped
sisa$WomPreg[is.na(sisa$WomPreg)]<-4
sisa3<-sisa[c(2:62,65:66,68:245,247:278,280:317,319:353,355:486,488:543),c(1:18,20:30)]#missing vars
anyNA(sisa3)

#this creates dummy variables automatically
dsisa3<-dummyVars(~.,data=sisa3,fullRank=TRUE)
dsisa<-data.frame(predict(dsisa3,newdata=sisa3))

#variables with zero or near zero variance are not useful to us
#check for nearzero vars
nzv <- nearZeroVar(dsisa, saveMetrics= TRUE)
nzv

#very highly correlated variables are also not useful to us
descrCor <-  cor(dsisa)
highCorr <- sum(abs(descrCor[upper.tri(dsisa)]) > .999)
highCorr

#linear combinations of variables that equal other variables are 
#also not useful
#this code will detect your dummy variables so ignore that
comboInfo <- findLinearCombos(dsisa)
comboInfo

##################################################
# step 2: create test and training sets
##################################################

#create training set
#you always have to set a seed if you want to get the same split
set.seed(1984)
sis.part <- createDataPartition(y = sisa3$Hospitalized,
                                ## the outcome data need to be specified
                                p = .85,
                                ## The percentage of data in the training set
                                list = FALSE)
sis.train<-sisa3[sis.part,]
sis.test<-sisa3[-sis.part,]

##################################################
# step 3: set fit controls to determine how best
#         algorithm is chosen, some algorithms 
#         need additional controls (tuning grid)
##################################################

fitControl <- trainControl(
    method = "repeatedcv",## 10-fold CV
    repeats = 10,## repeated ten times
    classProbs=TRUE,
    summaryFunction=twoClassSummary)

##################################################
# step 4: train all your algorithms
##################################################

####################logistic regression

set.seed(1984)#ensure that the same resamples are used
lr.sis <- train(hosp ~ ., data = sis.train, 
                method = "glm",
                family="binomial",
                trControl=fitControl,
                metric="ROC")

#mean accuracy and kappa from CV training
#this is your modeling object, created from all the training data
lr.sis 

#influence of variables
lrImp<-varImp(lr.sis,scale=FALSE)
plot(lrImp,main="Logistic Regression")

#####################neural networks, parameters: size, decay

#neural networks has tuning grid to fit other parameters 
nnetGrid <- expand.grid(size = c(1,3,5,10), #test these values for size
                        decay = c(0,0.1,0.25,0.5)) #test these for decay

set.seed(1984)#ensure that the same resamples are used
nnet.sis <- train(hosp ~ ., data = sis.train, 
                  method = "nnet",
                  tuneGrid=nnetGrid,
                  verbose=FALSE,
                  trControl=fitControl,
                  metric="ROC")

#mean accuracy and kappa from CV training
nnet.sis

#influence of variables
nnetImp<-varImp(nnet.sis,scale=FALSE)
plot(nnetImp,main="Neural Network")

#####################bagged trees

set.seed(1984)
bag.sis <- train(hosp ~ ., data = sis.train, 
                 method = "treebag",
                 trControl=fitControl,
                 verbose = FALSE,
                 metric="ROC")

#mean accuracy and kappa from CV training
bag.sis

#influence of variables
bagImp<-varImp(bag.sis,scale=FALSE)
plot(bagImp,main="Bagged Trees")

#####################k-nearest neighbors, parameter: k

#tuning grid for parameter fit
knnGrid <-  expand.grid(k = c(1:15))#test 1 to 15 neighbors

set.seed(1984)
knn.sis <- train(hosp ~ ., data = sis.train, 
                 method = "knn",
                 trControl=fitControl,
                 tuneGrid=knnGrid,
                 metric="ROC")

#mean accuracy and kappa from CV training
knn.sis

#influence of variables
knnImp<-varImp(knn.sis,scale=FALSE)
plot(knnImp,main="K-Nearest Neighbors")

####################random forest, parameter: mtry

#tuning grid for parameter 
rfGrid <-  expand.grid(mtry = c(1:12))

set.seed(1984)
rf.sis <- train(hosp ~ ., data = sis.train, 
                method = "rf",
                trControl=fitControl,
                verbose = FALSE,
                tuneGrid=rfGrid,
                importance=TRUE,
                metric="ROC")

#mean accuracy and kappa from CV training
rf.sis

#influence of variables
rfImp<-varImp(rf.sis,scale=FALSE)
plot(rfImp,main="Random Forest")

####################elastic net, parameters: lambda, fraction

#tuning grid for parameters 
enetGrid <- expand.grid(alpha = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                        lambda = c(0,0.001,0.01,0.1,0.25,0.5,1,2,2.5,3))

set.seed(1984)
enet.sis <- train(hosp ~ ., data = sis.train, 
                  method = "glmnet",
                  trControl=fitControl,
                  tuneGrid=enetGrid,
                  metric="ROC")

#mean accuracy and kappa from CV training
enet.sis

#influence of variables
enetImp<-varImp(enet.sis,scale=FALSE)
plot(enetImp,main="Elastic Net Regression")

####################generalized boosting models, 
####################parameters: interaction.depth, n.trees,
####################shrinkage, n.minobsinnode

#tuning grid for parameters 
gbmGrid <-  expand.grid(interaction.depth = c(1,5,9), 
                        n.trees=(1:5)*30, 
                        shrinkage=0.1,
                        n.minobsinnode=10)#make n.mino<10 for small trainings

set.seed(1984)
gbm.sis <- train(hosp~.,data=sis.train,
                 method = "gbm",
                 trControl=fitControl,
                 tuneGrid=gbmGrid,
                 verbose=FALSE,
                 metric="ROC")

#mean accuracy and kappa from CV training
gbm.sis

#influence of variables
gbmImp<-varImp(gbm.sis,scale=FALSE)
plot(gbmImp,main="Generalized Boosting Model")

##################################################
# step 5: use your trained algorithms (saved as 
#         xxx.sis) to make predictions on your
#         test set
##################################################

#make predictions and save
bagPred<-predict(bag.sis,newdata=sis.test)
knnPred<-predict(knn.sis,newdata=sis.test)
rfPred<-predict(rf.sis,newdata=sis.test)
gbmPred<-predict(gbm.sis,newdata=sis.test)
enetPred<-predict(enet.sis,newdata=sis.test)
nnetPred<-predict(nnet.sis,newdata=sis.test)
lrPred<-predict(lr.sis,newdata=sis.test)

#calculate accuracy and kappa for predictions
#versus observations in test set
postResample(pred=bagPred,obs=sis.test$hosp)
postResample(pred=knnPred,obs=sis.test$hosp)
postResample(pred=rfPred,obs=sis.test$hosp)
postResample(pred=gbmPred,obs=sis.test$hosp)
postResample(pred=enetPred,obs=sis.test$hosp)
postResample(pred=nnetPred,obs=sis.test$hosp)
postResample(pred=lrPred,obs=sis.test$hosp)

#calculate AUC for predictions versus observations 
#in test set

library(pROC)

sis.test$predn<-predict(nnet.sis,sis.test)
nnord<-factor(sis.test$pred,ordered=TRUE,levels=c("yes","no"))
obj.roc <- roc(sis.test$hosp,nnord )

sis.test$predb<-predict(bag.sis,sis.test)
bord<-factor(sis.test$predb,ordered=TRUE,levels=c("yes","no"))
b.roc <- roc(sis.test$hosp,bord )

sis.test$predk<-predict(knn.sis,sis.test)
kord<-factor(sis.test$predk,ordered=TRUE,levels=c("yes","no"))
k.roc <- roc(sis.test$hosp,kord )

sis.test$predr<-predict(rf.sis,sis.test)
rord<-factor(sis.test$predr,ordered=TRUE,levels=c("yes","no"))
r.roc <- roc(sis.test$hosp,rord )

sis.test$prede<-predict(enet.sis,sis.test)
eord<-factor(sis.test$prede,ordered=TRUE,levels=c("yes","no"))
e.roc <- roc(sis.test$hosp,eord )

sis.test$predg<-predict(gbm.sis,sis.test)
gord<-factor(sis.test$predg,ordered=TRUE,levels=c("yes","no"))
g.roc <- roc(sis.test$hosp,gord )

sis.test$predl<-predict(lr.sis,sis.test)
lord<-factor(sis.test$predl,ordered=TRUE,levels=c("yes","no"))
l.roc <- roc(sis.test$hosp,lord )
