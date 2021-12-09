################################################################
# Capstone: Chose Your Own project, Wine Wine Quality Data Set
################################################################

# Note: this process could take a couple of minutes
# (Sometimes it takes several tens of minutes)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(doParallel)){install.packages("doParallel", repos = "http://cran.us.r-project.org")} 
if(!require(xgboost)){install.packages("xgboost", repos = "http://cran.us.r-project.org")} 
if(!require(patchwork)){install.packages("patchwork", repos = "http://cran.us.r-project.org")} 
if(!require(dslabs)){install.packages("dslabs", repos = "http://cran.us.r-project.org")}
if(!require(torch)){install.packages("torch", repos = "http://cran.us.r-project.org")}
if(!require(tabnet)){install.packages("tabnet", repos = "http://cran.us.r-project.org")}
if(!require(tidymodels)){install.packages("tidymodels", repos = "http://cran.us.r-project.org")}
if(!require(withr)){install.packages("withr", repos = "http://cran.us.r-project.org")}
if(!require(vip)){install.packages("vip", repos = "http://cran.us.r-project.org")}
if(!require(recipes)){install.packages("recipes", repos = "http://cran.us.r-project.org")}
if(!require(rpart)){install.packages("rpart", repos = "http://cran.us.r-project.org")}
if(!require(rattle)){install.packages("rattle", repos = "http://cran.us.r-project.org")}
if(!require(kernlab)){install.packages("kernlab", repos = "http://cran.us.r-project.org")}
if(!require(ggcorrplot)){install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")}


# Loading all needed libraries

#library(doParallel) #Some libraries calculate in parallel, so we did not use them this time.
library(xgboost)
library(dslabs)
library(tidyverse)
library(caret)
library(knitr)
library(patchwork)
library(ggplot2)
library(data.table)
library(torch)
library(tabnet)
library(tidymodels)
library(withr)
library(recipes)
library(vip)
library(rpart)
library(rattle)
library(kernlab)
library(ggcorrplot)

# set the Parallel (to improve the speed)
# !!NO WORK During torch!!
#nc <- detectCores()
#cl <- makePSOCKcluster(nc)
#registerDoParallel(cl)

# Wine Quality Data Set
# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

url <- c("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
wine <- read_delim(url,delim = ";")

# modify the column name

colnames(wine) <- c("fixed_acidity", "volatile_acidity","citric_acid",
                    "residual_sugar","chlorides","free_sulfur_dioxide",
                    "total_sulfur_dioxide","density","pH",
                    "sulphates","alcohol","quality")

### Check the data -----

# whole the data

head(wine) %>% kable()
wine %>% summary()
sum(is.na(wine))

# Target: quality

table(wine$quality) 

# Distribution of each value

wine %>% gather() %>% 
  ggplot(aes(value)) + 
  geom_histogram(col="black") +
  facet_wrap(~key, scales = "free")

# Check the relations between each attribute and quality (target)

wine  %>% reshape2::melt(.,"quality") %>% 
  ggplot(aes(value, quality)) + 
  geom_point() +
  geom_smooth(method = lm, col = "red", fill = "red") +
  facet_wrap(~variable, scales = "free")

# Check the correlation 
## Compute a matrix of correlation p-values

p.mat <- cor_pmat(wine)

# X means no significant coefficient

round(cor(wine),2) %>% 
ggcorrplot(., method = "circle",  hc.order = TRUE, type = "lower", p.mat = p.mat)


### Data preparation: Split the data set ----
# Preparation test and train data set for model selection
# Due to the small amount of data, 20% was used as test data.

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = wine$quality, times = 1, p = 0.2, list = FALSE)
train_set <- wine[-test_index,]
test_set <- wine[test_index,]

# To build each model, centralization and standardization are done with recipe.

rec_pre <- recipe(train_set, quality~.) %>% 
  step_center(all_numeric(),-quality) %>% # Centralization 
  step_scale(all_numeric(), -quality) %>%  # Standardization
  # step_knnimpute(all_predictors(),K=5)  # There is no NA
  # step_dummy(all_predictors(), -all_numeric()) # There is no categorical data
  prep()

# apply the recipe

train <- bake(rec_pre, new_data = train_set)
test <-  bake(rec_pre, new_data = test_set)

# check the data

head(train) %>% kable()
head(test) %>% kable()

# Transform "quality" into binary

train$quality[train$quality < 6] <- 0
test$quality[test$quality < 6] <- 0
train$quality[train$quality > 0] <- 1
test$quality[test$quality > 0] <- 1

train$quality <- as.factor(train$quality)
test$quality <- as.factor(test$quality)


# summary 

summary(train)
summary(test)



#################
# MODEL Analysis
#################

# set the Cross Validation for caret package

trControl = trainControl(method = "cv", number = 10)

# logistic ------

set.seed(1, sample.kind="Rounding")

# train the model

wine_logit <- train(
  quality ~ .,
  data = train,
  method = "glm",
  family = binomial(),
  trControl = trControl
)

# Check the model

summary(wine_logit)
p_logit <- varImp(wine_logit,scale = F) %>% 
  ggplot() + ggtitle("Logit")
p_logit

# Evaluate

confusionMatrix(predict(wine_logit, test), test$quality)
logit_acc <- confusionMatrix(predict(wine_logit, test), test$quality)$overall["Accuracy"]
logit_acc

### Rpart -------

set.seed(1, sample.kind="Rounding")

# train the model

wine_rpart <- train(
  quality ~ .,
  data = train,
  method = "rpart",
  trControl = trControl,
  tuneLength = 10
)

# Check the model

wine_rpart
fancyRpartPlot(wine_rpart$finalModel)
p_rpart <- varImp(wine_rpart,scale = F) %>% 
  ggplot() + ggtitle("Rpart")
p_rpart


# Evaluate

confusionMatrix(predict(wine_rpart, test), test$quality)
rpart_acc <- confusionMatrix(predict(wine_rpart, test), test$quality)$overall["Accuracy"]
rpart_acc


### Random forest ------

set.seed(1, sample.kind="Rounding")

# train the model

wine_rf <- train(
  quality ~ .,
  data = train,
  method = "rf",
  trControl = trControl,
  tuneLength = 10
)

# Check the model

wine_rf
ggplot(wine_rf)
p_rf <- varImp(wine_rf,scale = F) %>% 
  ggplot() + ggtitle("RF")
p_rf


# Evaluate

confusionMatrix(predict(wine_rf, test), test$quality)
rf_acc <- confusionMatrix(predict(wine_rf, test), test$quality)$overall["Accuracy"]
rf_acc


### SVM ------


set.seed(1, sample.kind="Rounding")

# train the model

wine_svm <- train(
  quality ~ .,
  data = train,
  method = "svmRadial",
  trControl = trControl,
  tuneLength = 10
)

# Check the model

wine_svm
ggplot(wine_svm)


# Evaluate

confusionMatrix(predict(wine_svm, test), test$quality)
svm_acc <- confusionMatrix(predict(wine_svm, test), test$quality)$overall["Accuracy"]
svm_acc



### Neural Net ------

set.seed(1, sample.kind="Rounding")

# train the model

wine_nnet <- train(
  quality ~ .,
  data = train,
  method = "nnet",
  trControl = trControl,
  tuneLength = 10,
  trace = FALSE
)

# Check the model

wine_nnet
ggplot(wine_nnet)
p_nnet <- varImp(wine_nnet,scale = F) %>% 
  ggplot() + ggtitle("NNet")
p_nnet


# Evaluate

confusionMatrix(predict(wine_nnet, test), test$quality)
nnet_acc <- confusionMatrix(predict(wine_nnet, test), test$quality)$overall["Accuracy"]
nnet_acc


### XGBoost: xgbLinear----

set.seed(1, sample.kind="Rounding")

# train the model

wine_xgbl <- train(
  quality ~ .,
  data = train,
  method = "xgbLinear",
  trControl = trControl,
  tuneLength = 5
)

# Check the model

wine_xgbl
ggplot(wine_xgbl)
p_xgbl <- varImp(wine_xgbl,scale = F) %>% 
  ggplot() + ggtitle("XGB L")
p_xgbl


# Evaluate

confusionMatrix(predict(wine_xgbl, test), test$quality)
xgbl_acc <- confusionMatrix(predict(wine_xgbl, test), test$quality)$overall["Accuracy"]
xgbl_acc



### XGBoost: xgbDART----

set.seed(1, sample.kind="Rounding")

# train the model

wine_xgbd <- train(
  quality ~ .,
  data = train,
  method = "xgbDART",
  trControl = trControl,
  tuneLength = 3
)

# Check the model

wine_xgbd
p_xgbd <- varImp(wine_xgbd,scale = F) %>% 
  ggplot() + ggtitle("XGB D")
p_xgbd


# Evaluate

confusionMatrix(predict(wine_xgbd, test), test$quality)
xgbd_acc <- confusionMatrix(predict(wine_xgbd, test), test$quality)$overall["Accuracy"]
xgbd_acc




### XGBoost: xgbTree----

set.seed(1, sample.kind="Rounding")

# train the model

wine_xgbt <- train(
  quality ~ .,
  data = train,
  method = "xgbTree",
  trControl = trControl,
  tuneLength = 5
)

# Check the model

wine_xgbt
p_xgbt <- varImp(wine_xgbt,scale = F) %>% 
  ggplot() + ggtitle("XGB T")
p_xgbt


# Evaluate

confusionMatrix(predict(wine_xgbt, test), test$quality)
xgbt_acc <- confusionMatrix(predict(wine_xgbt, test), test$quality)$overall["Accuracy"]
xgbt_acc


### Torch / Deep Learning -----

# Change the type from factor to numeric for DNN
train_torch <- 
  train %>% mutate(
    quality = as.numeric(quality) - 1
    )

test_torch <- 
  test %>% mutate(
    quality = as.numeric(quality) - 1
  )

# check the quality 

table(train$quality,train_torch$quality)

# Create {torch} dataset

df_dataset <- dataset(
  
  "wine",
  
  initialize = function(df, response_variable) {
    self$df <- df[,-which(names(df) == response_variable)]
    self$response_variable <- df[[response_variable]]
  },
  
  .getitem = function(index) {
    response <- torch_tensor(self$response_variable[index])
    x <- torch_tensor(as.numeric(self$df[index,]))
    
    list(x = x, y = response)
  },
  
  .length = function() {
    length(self$response_variable)
  }
  
)

# Create the data set train and test

train_torch_ds <- df_dataset(train_torch, "quality")
test_torch_ds <- df_dataset(test_torch, "quality")

# Create the data loader train and test

train_torch_dl <- dataloader(train_torch_ds, batch_size = 32, shuffle = T)
test_torch_dl <- dataloader(test_torch_ds, batch_size = 1, shuffle = F)

# Define a network / fc1 - 3 and dropout

net <- nn_module(
  
  "wine_DNN",
  
  initialize = function() {
    self$fc1 <- nn_linear(11, 66)
    self$fc2 <- nn_linear(66, 44)
    self$fc3 <- nn_linear(44, 1)
    self$dropout <- nn_dropout(0.5)
  },
  
  forward = function(x) {
    x %>% 
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$dropout() %>%
      self$fc3() %>%
      nnf_sigmoid()
  }
)

# Use CPU version

model <- net()
model$to(device = "cpu")

# Set the optimizer condition

optimizer <- optim_adam(model$parameters,lr = 0.01)

# Run a learning iteration

coro::loop(for (epoch in 1:20) {
  
  l <- c()
  
  coro::loop(for (b in enumerate(train_torch_dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]]$to(device = "cpu"))
    loss <- nnf_binary_cross_entropy_with_logits(output, b[[2]]$to(device = "cpu"))
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  })
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
})

# Model prediction

model$eval()

i <- 1
pred_labels <- rep(0, nrow(test_torch))

coro::loop(for (b in enumerate(test_torch_dl)) {
  output <- model(b[[1]]$to(device = "cpu"))
  pred_labels[i] <- round(output$item(), 0)
  i <- i + 1
})

# Evaluate

table(test=test_torch$quality, pred_labels)

torch_acc <- sum(diag(table(test_torch$quality, pred_labels))) / nrow(test_torch)
torch_acc


### Tabnet -----

# splits the data (set vfold to 4 because it takes time)

splits <-  
  vfold_cv(train, v = 4, strata = "quality") %>% 
  with_seed(1, .)

# Create rule "classification"

rule <- tabnet(
  epochs = tune(), 
  penalty = tune(),
  batch_size = tune(), 
  decision_width = tune(), 
  attention_width =tune(),
  num_steps = tune(),
  learn_rate = 0.08,
  momentum = 0.6) %>%
  set_engine("torch", verbose = TRUE) %>% 
  set_mode("classification")

# making a recipe

rec <- recipe(quality ~ ., data = train) %>% 
  step_nzv(all_predictors())

# making a work flow

wf <- workflow() %>% 
  add_model(rule) %>% 
  add_recipe(rec) 

# range of hyper parameter 

range_hypara <- 
  wf %>% 
  parameters() %>% 
  update(
    epochs = epochs(c(50, 70)),
    decision_width = decision_width(range = c(20, 40)),
    attention_width = attention_width(range = c(20, 40)),
    num_steps = num_steps(range = c(4, 8))
  ) %>% 
  finalize(train)

# making a grid

grid <- range_hypara %>% 
  grid_latin_hypercube(size = 1) %>% 
  with_seed(1, .)


# tuning of hyper parameters (long time)

tune <-  
  wf %>% 
  tune_grid(
    resamples = splits,
    grid = grid,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(accuracy)
  ) 

# select the best hyper parameters 

good_hypara <-
  tune %>% 
  show_best() %>% 
  dplyr::slice(1)

# update the rule with best parameters

upd_rule <- 
  tabnet(
    epochs = good_hypara %>% pull(epochs), 
    penalty = good_hypara %>% pull(penalty),
    batch_size = good_hypara %>% pull(batch_size), 
    decision_width = good_hypara %>% pull(decision_width), 
    attention_width = good_hypara %>% pull(attention_width),
    num_steps = good_hypara %>% pull(num_steps),
    learn_rate = 0.08,
    momentum = 0.6) %>%
  set_engine("torch", verbose = TRUE) %>%
  set_mode("classification")


# update the work flow with updated rule

upd_wf <- 
  workflow() %>% 
  add_model(upd_rule) %>% 
  add_recipe(rec)

# build the model

model_tn <- 
  upd_wf %>% 
  fit(train) %>% 
  with_seed(1,.)


# Predict 

pred_tabnet <- 
  predict(model_tn,new_data = test, type = "class")


# Evaluate 

table(test=test$quality, pred= pred_tabnet$.pred_class)

tabnet_acc <- sum(diag(table(test$quality, pred_tabnet$.pred_class))) /
    nrow(test)

tabnet_acc

# Check the importance

fit <- extract_fit_parsnip(model_tn)
p_tabnet <- vip(fit) + ggtitle("Tabnet") 
p_tabnet

### Compare the models -----

rbind(logit_acc, rpart_acc, rf_acc, svm_acc, nnet_acc,
      xgbl_acc, xgbd_acc, xgbt_acc, torch_acc, tabnet_acc) %>% kable()


wrap_plots(p_logit, p_rpart, p_nnet, p_rf, p_xgbl, p_xgbd, p_xgbt, p_tabnet)
