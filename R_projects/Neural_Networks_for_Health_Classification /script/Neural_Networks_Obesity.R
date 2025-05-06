# load library
library(GGally)
library(nnet) 
library(e1071)
library(caret)
library(dplyr)



# load data file
df = read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# data explore
str(df)
head(df)

# plot numerical variables
numeric_vars <- which(sapply(df, is.numeric))

ggpairs(
  data = df,
  columns = numeric_vars,
  mapping = aes(colour = NObeyesdad, alpha = 0.4),
  upper = list(
    continuous = wrap("cor", size = 2)  
  )
) +
  theme(
    text = element_text(size = 10)       
  )


# plot categorical variables
a <- ggplot(df, aes(x=Gender, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="Gender", y="Frequency") + 
  theme(legend.position = "none")
b <- ggplot(df, aes(x=family_history_with_overweight, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="Family History", y="Frequency") + 
  theme(legend.position = "none")
c <- ggplot(df, aes(x=FAVC, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="FAVC", y="Frequency") + 
  theme(legend.position = "none")
d <- ggplot(df, aes(x=CAEC, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="CAEC", y="Frequency") + 
  theme(legend.position = "none")
e <- ggplot(df, aes(x=SMOKE, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="SMOKE", y="Frequency") + 
  theme(legend.position = "none")
f <- ggplot(df, aes(x=SCC, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="SCC", y="Frequency") + 
  theme(legend.position = "none")
g <- ggplot(df, aes(x=CALC, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="CALC", y="Frequency") + 
  theme(legend.position = "none")
h <- ggplot(df, aes(x=MTRANS, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="MTRANS", y="Frequency") + 
  theme(legend.position = "none")
i <- ggplot(df, aes(x=NObeyesdad, fill=NObeyesdad)) + 
  geom_bar(position = "stack") + 
  labs(x="NObeyesdad", y="Frequency") + 
  theme(legend.position = "none")

grid.arrange(arrangeGrob(a, b, c, d, e, f, g, h, i, 
                         ncol=5, nrow = 2))

# convert all categorical variables to factor
#chr_vars <- sapply(df, is.character)
#df[chr_vars] <- lapply(df[chr_vars], as.factor)

#dummy_vars <- dummyVars(~ ., data = df[, -17])
#df_encoded <- predict(dummy_vars, newdata = df)

# drop all categorical variables only keep numerical variables 
df_numerical <- df[, sapply(df, is.numeric) | names(df) == "NObeyesdad"]
df_numerical$NObeyesdad <- as.factor(df_numerical$NObeyesdad)

# stratified spliting, 75% training and 25% test
set.seed(1234)

train_indx <- createDataPartition(df_numerical$NObeyesdad, p = 0.75, list = FALSE)
train_data <- df_numerical[train_indx, ]
test_data <-df_numerical[-train_indx, ]

# check the class distribution in training and test sets
prop.table(table(train_data$NObeyesdad))*100
prop.table(table(test_data$NObeyesdad))*100

# build a base model
cls <- class.ind(train_data$NObeyesdad)
nn_model_base <- nnet(
  x = train_data[, -which(names(train_data) == "NObeyesdad")], 
  y = cls,                                                  
  size = 10,
  decay = 2,
  softmax = TRUE)

#Apply to the test set
nn_base_pred<-predict(nn_model_base, test_data[, -which(names(train_data) == "NObeyesdad")], type="class")
tab<-table(test_data$NObeyesdad,nn_base_pred)

classAgreement(tab)

## optimize size and decay rate using cross-validation
set.seed(1234)
tune_parameter = tune.nnet(NObeyesdad~., data = train_data, size = 1:20,decay=0:5,tunecontrol = tune.control(sampling = "cross",cross=5))
summary(tune_parameter)
plot(tune_parameter)

set.seed(1234)
tune_parameter_1 = tune.nnet(NObeyesdad~., data = train_data, size = 1:20,decay=0:8,tunecontrol = tune.control(sampling = "cross",cross=10))
summary(tune_parameter_1)
plot(tune_parameter_1)


## build optimize model
nn_model_opt <- nnet(
  x = train_data[, -which(names(train_data) == "NObeyesdad")], 
  y = cls,                                                  
  size = 19,
  decay = 0,
  softmax = TRUE)

#Apply to the test set
nn_opt_pred<-predict(nn_model_opt, test_data[, -which(names(train_data) == "NObeyesdad")], type="class")
tab_opt<-table(test_data$NObeyesdad,nn_opt_pred)

classAgreement(tab_opt)

## build second optimize model
nn_model_opt_1 <- nnet(
  x = train_data[, -which(names(train_data) == "NObeyesdad")], 
  y = cls,                                                  
  size = 20,
  decay = 0,
  softmax = TRUE)

#Apply to the test set
nn_opt_pred_1<-predict(nn_model_opt_1, test_data[, -which(names(train_data) == "NObeyesdad")], type="class")
tab_opt_1<-table(test_data$NObeyesdad,nn_opt_pred_1)

classAgreement(tab_opt_1)

# variable importance
# Compute variable importance
var_imp <- olden(nn_model_opt_1, bar_plot = TRUE)

# Visualize importance
plot(var_imp)

#======================================
# scale train_data
train_means <- apply(train_data[, sapply(train_data, is.numeric)], 2, mean)
train_sds <- apply(train_data[, sapply(train_data, is.numeric)], 2, sd)

train_data_scaled <- train_data
train_data_scaled[, sapply(train_data, is.numeric)] <- scale(
  train_data[, sapply(train_data, is.numeric)],
  center = train_means,
  scale = train_sds
)

# scale test_data using training data parameters
test_data_scaled <- test_data
test_data_scaled[, sapply(test_data, is.numeric)] <- scale(
  test_data[, sapply(test_data, is.numeric)],
  center = train_means,
  scale = train_sds
)

# target variable remains unchanged
train_data_scaled$NObeyesdad <- train_data$NObeyesdad
test_data_scaled$NObeyesdad <- test_data$NObeyesdad


set.seed(1234)
tune_parameter_2 = tune.nnet(NObeyesdad~., data = train_data_scaled, size = 1:20,decay=0:8,tunecontrol = tune.control(sampling = "cross",cross=10))
summary(tune_parameter_2)
plot(tune_parameter_2)

cls_1 <- class.ind(train_data_scaled$NObeyesdad)
## build optimize model-3
nn_model_opt_2 <- nnet(
  x = train_data_scaled[, -which(names(train_data) == "NObeyesdad")], 
  y = cls_1,                                                  
  size = 8,
  decay = 0,
  softmax = TRUE)

#Apply to the test set
nn_opt_pred_2<-predict(nn_model_opt_2, test_data_scaled[, -which(names(train_data) == "NObeyesdad")], type="class")
tab_opt_2<-table(test_data_scaled$NObeyesdad,nn_opt_pred_2)

classAgreement(tab_opt_2)






