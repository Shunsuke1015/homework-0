##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           # title = as.character(title),
                                           # genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



### Quiz ----------------
# Q1
nrow(edx)
ncol(edx)
# Q2
edx %>% select(rating) %>% table()
# Q3
edx %>% select(movieId) %>% unique() %>% nrow()
# Q4
n_distinct(edx$userId)

# Q5
edx %>% group_by(genres) 

filter(edx, str_detect(genres, "Drama")) %>% nrow()
filter(edx, str_detect(genres, "Comedy")) %>% nrow()
filter(edx, str_detect(genres, "Thriller")) %>% nrow()
filter(edx, str_detect(genres, "Romance")) %>% nrow()

# Q6
edx %>% group_by(title) %>% summarize(number=n()) %>%  arrange(desc(number))

# Q7
edx %>% group_by(rating) %>% summarize(count=n()) %>% arrange(desc(count))

# Q8
edx %>% group_by(rating) %>% summarize(count=n())


edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()








##########################################################
# MovieLens Reccomender System Project
##########################################################

# Install all needed libraries if it is not present
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(knitr)) install.packages("knitr")
if(!require(recosystem)){install.packages("recosystem")} 
if(!require(doParallel)){install.packages("doParallel")} 
if(!require(xgboost)){install.packages("xgboost")} 
if(!require(patchwork)){install.packages("patchwork")} 
if(!require(caret)){install.packages("caret")}
if(!require(dslabs)){install.packages("dslabs")}
# if(!require()){install.packages("")} 
# if(!require()){install.packages("")} 


# Loading all needed libraries
library(doParallel)
library(xgboost)
library(dslabs)
library(tidyverse)
library(caret)
library(recosystem)
library(knitr)
library(patchwork)
library(ggplot2)


# set the Parallel (to improve the speed)
nc <- detectCores()
cl <- makePSOCKcluster(nc)
registerDoParallel(cl)

# Define Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Convert timestamp to a human readable date
edx$date <- as.POSIXct(edx$timestamp, origin="1970-01-01")
validation$date <- as.POSIXct(validation$timestamp, origin="1970-01-01")

# Extract the year of Rate in both data sets

edx$year_Rate <- as.integer(format(edx$date,"%Y"))
validation$year_Rate <- as.integer(format(validation$date,"%Y"))


# Extract the year of release for each movie in both data set
# edx dataset
edx <- edx %>%
  mutate(title = str_trim(title)) %>%
  extract(title,
          c("titleTemp", "release"),
          regex = "^(.*) \\(([0-9 \\-]*)\\)$",
          remove = F) %>%
  mutate(release = if_else(str_length(release) > 4,
                           as.integer(str_split(release, "-",
                                                simplify = T)[1]),
                           as.integer(release))
  ) %>%
  mutate(title = if_else(is.na(titleTemp),
                         title,
                         titleTemp)
  ) %>%
  select(-titleTemp)

# validation data set
validation <- validation %>%
  mutate(title = str_trim(title)) %>%
  extract(title,
          c("titleTemp", "release"),
          regex = "^(.*) \\(([0-9 \\-]*)\\)$",
          remove = F) %>%
  mutate(release = if_else(str_length(release) > 4,
                           as.integer(str_split(release, "-",
                                                simplify = T)[1]),
                           as.integer(release))
  ) %>%
  mutate(title = if_else(is.na(titleTemp),
                         title,
                         titleTemp)
  ) %>%
  select(-titleTemp)



# Preparation test and train data set for model selection
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]


# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# Data Exploaration
head(edx) %>% kable()
str(edx) 
dim(edx)
sum(is.na(edx))
summary(edx) %>% kable()
edx%>% summarize(n_users = n_distinct(userId),
                 n_movies = n_distinct(movieId),
                 n_genres = n_distinct(genres)) %>% kable()

# EDA
# Whole count distribution
p_c1 <- edx %>%  
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

p_c2 <- edx %>%  
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# compare count distribution by movie and user
p_c1 + p_c2

# Whole rating distribution
p_d1 <- edx %>% group_by(movieId) %>%
  summarise(mean = mean(rating)) %>% 
  ggplot(aes(mean)) + geom_histogram(bins = 25,col="black") +
  ggtitle("Rating distribution by Movie")

p_d2 <- edx %>% group_by(userId) %>%
  summarise(mean = mean(rating)) %>% 
  ggplot(aes(mean)) + geom_histogram(bins = 25,col="black") +
  ggtitle("Rating distribution by User")

# compare rating distribution by movie and user
p_d1 / p_d2

# rating distribution by year of Rating
edx %>%  ggplot(aes(rating)) + 
  geom_histogram(bins = 25,col="black") +
  facet_wrap(~year_Rate)


# Genres
edx %>% group_by(genres) %>% 
  summarise(n=n()) %>%
  head()

# Separate genres and rating distribution by genres
edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  select(genres, rating) %>% 
  group_by(genres) %>%
  summarize(count = n(), mean = mean(rating)) %>% kable()

# Identify trends in the time period being evaluated focusing ranking.
genresByYear <- edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  select(movieId, year_Rate, genres, rating) %>% 
  group_by(year_Rate, genres) %>% 
  filter(!is.na(rating)) %>% 
  summarise(count = n(), 
            rating_avg = mean(rating)) %>%
  mutate(count_rank = row_number(desc(count)),
         rating_rank = row_number(rating_avg))

# Create the graph count ranking
p_genre1 <- genresByYear %>% #filter(year_Rate>=1993) %>% 
  ggplot(aes(x = year_Rate, y= count_rank, group=genres))+
  geom_line(aes(color = genres, alpha = 1), size = 2) +
  geom_point(aes(color = genres, alpha = 1), size = 4) +
  scale_y_reverse(breaks = 1:nrow(genresByYear)) +
  guides(color=guide_legend(ncol=2)) +
  ggtitle("Ranking of count by year of Rate")
p_genre1

# Create the graph rating ranking
p_genre2 <- genresByYear %>% #filter(year_Rate>=1993) %>% 
  ggplot(aes(x = year_Rate, y= rating_rank, group=genres))+
  geom_line(aes(color = genres, alpha = 1), size = 2) +
  geom_point(aes(color = genres, alpha = 1), size = 4) +
  scale_y_reverse(breaks = 1:nrow(genresByYear)) +
  guides(color=guide_legend(ncol=2)) +
  ggtitle("Ranking of Rating by year of Rate")
p_genre2


# Extract the genres name
genre_name <- unique(genresByYear$genres)

#################
# MODEL Analysis
#################

# Creating the Target
result <- data.frame(Method = "Target", RMSE = 0.8649)

# Initial Prediction-----
# Mean of observed values
mu <- mean(train_set$rating)

# Update the result table  
result <- bind_rows(result, data.frame(Method = "Mean", RMSE = RMSE(test_set$rating, mu)))

# Show the RMSE
result %>% kable()

# Add Movie Effect(bi) -----
# Movie effects (bi)
bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
head(bi) %>% kable()

# Confirm the Movie effects distribution
bi %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=25, col = I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie effect") +
  ylab("Count") 

# Predict the rating with mean + bi  
y_hat_bi <- mu + test_set %>% 
  left_join(bi, by = "movieId") %>% pull(b_i)


# Calculate the RMSE and update the result
result <- bind_rows(result, 
                    data.frame(Method = "Mean + bi", RMSE = RMSE(test_set$rating, y_hat_bi)))

# Show the RMSE improvement  
result %>% kable()


# Add User Effect(bu) ------
# User effect (bu)
bu <- train_set %>% 
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Confirm the User effects distribution
bu %>% # filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins=25,color = "black") + 
  ggtitle("User Effect Distribution") +
  xlab("User Effect") +
  ylab("Count")

# Prediction
y_hat_bi_bu <- test_set %>% 
  left_join(bi, by="movieId") %>%
  left_join(bu, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

# Calculate the RMSE and update result
result <- bind_rows(result, 
                    data.frame(Method = "Mean + bi + bu", RMSE = RMSE(test_set$rating, y_hat_bi_bu)))

# Show the RMSE improvement  
result %>% kable()

# Add year of Rate Effect(by) ------
# year of Rate effect (by)
by <- train_set %>% 
  left_join(bi, by = "movieId")%>%
  left_join(bu, by="userId") %>%
  group_by(year_Rate) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

# Confirm the year of Rate effects distribution
by %>% 
  ggplot(aes(b_y)) + 
  geom_histogram(bins=100,color = "black") + 
  ggtitle("Year of Rate Effect Distribution") +
  xlab("Year of Rate Effect") +
  ylab("Count")

# Prediction
y_hat_bi_bu_by <- test_set %>% 
  left_join(bi, by="movieId") %>%
  left_join(bu, by="userId") %>%
  left_join(by, by="year_Rate") %>% 
  mutate(pred = mu + b_i + b_u + b_y) %>% pull(pred)

# Calculate the RMSE and update result
result <- bind_rows(result, 
                    data.frame(Method = "Mean + bi + bu + by",
                               RMSE = RMSE(test_set$rating, y_hat_bi_bu_by)))

# Show the RMSE improvement  
result %>% kable()

# Add Genre Effect(bg) ------
# Genre Rate effect (bg)
bg <- train_set %>% 
  left_join(bi, by = "movieId")%>%
  left_join(bu, by="userId") %>%
  left_join(by, by="year_Rate") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i- b_u - b_y))

# Confirm the Genre effects distribution
bg %>% 
  ggplot(aes(b_g)) + 
  geom_histogram(bins=25,color = "black") + 
  ggtitle("Genre Effect Distribution") +
  xlab("Genre Effect") +
  ylab("Count")

# Prediction
y_hat_bi_bu_by_bg <- test_set %>% 
  left_join(bi, by="movieId") %>%
  left_join(bu, by="userId") %>%
  left_join(by, by="year_Rate") %>%
  left_join(bg, by="genres") %>% 
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>% pull(pred)

# Calculate the RMSE and update result
result <- bind_rows(result, 
                    data.frame(Method = "Mean + bi + bu + by + bg",
                               RMSE = RMSE(test_set$rating, y_hat_bi_bu_by_bg)))

# Show the RMSE improvement  
result %>% kable()



# Regularization----
# Movie + User (Same methods the course)
lambdas <- seq(0, 10, 0.2)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  bi <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  bu <- train_set %>% 
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Check the lambdas
qplot(lambdas,rmses)

lambda <- lambdas[which.min(rmses)]
lambda

# Calculate the RMSE and update result
result <- bind_rows(result, 
                    data.frame(Method = "Regularized Movie + User Effect", RMSE = min(rmses)))

# Show the RMSE improvement  
result %>% kable()


# Movie + User + Year of Rate + Genre (new one, regularization with 4 biases)
lambdas <- seq(0, 10, 0.2)
rmses2 <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  bi <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  bu <- train_set %>% 
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  by <- train_set %>% 
    left_join(bi, by = "movieId")%>%
    left_join(bu, by="userId") %>%
    group_by(year_Rate) %>%  
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  bg <- train_set %>% 
    left_join(bi, by = "movieId")%>%
    left_join(bu, by="userId") %>%
    left_join(by, by="year_Rate") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(by, by="year_Rate") %>%
    left_join(bg, by="genres") %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Check the lambdas
qplot(lambdas,rmses2)

lambda <- lambdas[which.min(rmses2)]
lambda

# Calculate the RMSE and update result
result <- bind_rows(result, 
                    data.frame(Method = "Regularized Movie + User + Year of Rate + Genre Effect", RMSE = min(rmses2)))

# Show the RMSE improvement  
result %>% kable()




# Matrix Factorization-----
set.seed(1, sample.kind = "Rounding")

# Convert "train" and "test" sets to recosystem input format
train_reco <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating = rating))
test_reco  <-  with(test_set, data_memory(user_index = userId, 
                                          item_index = movieId, 
                                          rating = rating))

# Create the model object
reco <-  recosystem::Reco()

# Tune the parameters
opts <-  reco$tune(train_reco, opts = list(dim = c(10, 20, 30), 
                                           lrate = c(0.1, 0.2),
                                           costp_l2 = c(0.01, 0.1), 
                                           costq_l2 = c(0.01, 0.1),
                                           nthread  = nc, niter = 10))

# Train the model
reco$train(train_reco, opts = c(opts$min, nthread = nc, niter = 20))

# Calculate the prediction
y_hat_reco <-  reco$predict(test_reco, out_memory())

# Update the result table
result <- bind_rows(result, 
                    data.frame(Method = "Matrix Factorization - recosystem", 
                               RMSE = RMSE(test_set$rating, y_hat_reco)))

# Show the RMSE improvement  
result %>% kable()




# Final Validation-------
set.seed(1, sample.kind = "Rounding")

# Convert "edx" and "validation" sets to recosystem input format
edx_reco <-  with(edx, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating = rating))
valid_reco  <-  with(validation, data_memory(user_index = userId, 
                                          item_index = movieId, 
                                          rating = rating))

# Create the model object
reco_final <-  recosystem::Reco()

# Tune the parameters
opts_final <-  reco_final$tune(edx_reco, opts = list(dim = c(10, 20, 30), 
                                           lrate = c(0.1, 0.2),
                                           costp_l2 = c(0.01, 0.1), 
                                           costq_l2 = c(0.01, 0.1),
                                           nthread  = nc, niter = 10))

# Train the model
reco_final$train(edx_reco, opts = c(opts_final$min, nthread = nc, niter = 20))

# Calculate the prediction
y_hat_final_reco <-  reco_final$predict(valid_reco, out_memory())

# Update the result table
result <- bind_rows(result, 
                    data.frame(Method = "[FINAL model] Matrix Factorization - recosystem", 
                               RMSE = RMSE(validation$rating, y_hat_final_reco)))

# Show the RMSE improvement  
result %>% kable()


# Reference: What is the RMSE of regularized 4 effects
# With edx and validation data sets
# Movie + User + Year of Rate + Genre (new one, regularization with 4 biases)
lambda <- 5

# Attention: Validation data set cannot be used for select the model
rmse_v <- sapply(lambda, function(l){
  mu <- mean(edx$rating)
  bi <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  bu <- edx %>% 
    left_join(bi, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  by <- edx %>% 
    left_join(bi, by = "movieId")%>%
    left_join(bu, by="userId") %>%
    group_by(year_Rate) %>%  
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  bg <- edx %>% 
    left_join(bi, by = "movieId")%>%
    left_join(bu, by="userId") %>%
    left_join(by, by="year_Rate") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(by, by="year_Rate") %>%
    left_join(bg, by="genres") %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})


# Update result
result <- bind_rows(result, 
                    data.frame(Method = "[Reference] Regularized 4 Effects with Lambda=5", RMSE = rmse_v))

# Show the RMSE improvement  
result %>% kable()






# For Discussion
data("movielens")

# Preparation test and train data set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
train_small <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in test set are also in train set
test_small <- temp %>% 
  semi_join(train_small, by = "movieId") %>%
  semi_join(train_small, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_small)
train_small <- rbind(train_small, removed)

rm(test_index, temp, removed)

# To compare, targeted same value: 0.8649
result_small <- data.frame(Method = "Target", RMSE = 0.8649)

# Initial Prediction-----
# Mean of observed values
mu <- mean(train_small$rating)

# Update the result_small  
result_small <- bind_rows(result_small, data.frame(Method = "Mean", RMSE = RMSE(test_small$rating, mu)))

# Show the RMSE
result_small %>% kable()

# Add Movie Effect(bi) -----
# Movie effects (bi)
bi <- train_small %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predict the rating with mean + bi  
y_hat_bi <- mu + test_small %>% 
  left_join(bi, by = "movieId") %>% pull(b_i)


# Calculate the RMSE and update result_small
result_small <- bind_rows(result_small, 
                          data.frame(Method = "Mean + bi", RMSE = RMSE(test_small$rating, y_hat_bi)))

# Show the RMSE improvement
result_small %>% kable()


# Add User Effect(bu) ------
# User effect (bu)
bu <- train_small %>% 
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Prediction
y_hat_bi_bu <- test_small %>% 
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

# Calculate the RMSE and update result_small
result_small <- bind_rows(result_small, 
                          data.frame(Method = "Mean + bi + bu", RMSE = RMSE(test_small$rating, y_hat_bi_bu)))

# Show the RMSE improvement  
result_small %>% kable()

# Regularization----
lambdas <- seq(0, 10, 0.2)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_small$rating)
  b_i <- train_small %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_small %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_small %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_small$rating))
})

qplot(lambdas,rmses)

lambda <- lambdas[which.min(rmses)]
lambda

# Calculate the RMSE and update result_small
result_small <- bind_rows(result_small, 
                          data.frame(Method = "Regularized Movie + User Effect", RMSE = min(rmses)))

# Show the RMSE improvement  
result_small %>% kable()


# Matrix Factorization-----
set.seed(1, sample.kind = "Rounding")

# Convert 'train_small' and 'test_small' sets to recosystem input format
train_reco <-  with(train_small, data_memory(user_index = userId, 
                                             item_index = movieId, 
                                             rating = rating))
test_reco  <-  with(test_small, data_memory(user_index = userId, 
                                            item_index = movieId, 
                                            rating = rating))

# Create the model object
reco <-  recosystem::Reco()

# Tune the parameters
opts <-  reco$tune(train_reco, opts = list(dim = c(10, 20, 30), 
                                           lrate = c(0.1, 0.2),
                                           costp_l2 = c(0.01, 0.1), 
                                           costq_l2 = c(0.01, 0.1),
                                           nthread  = nc, niter = 10)) # nc is depended on PC

# Train the model
reco$train(train_reco, opts = c(opts$min, nthread = nc, niter = 20))

# Calculate the prediction
y_hat_final_reco <-  reco$predict(test_reco, out_memory())

# Update the result_small table
result_small <- bind_rows(result_small, 
                          data.frame(Method = "Matrix Factorization - recosystem", 
                                     RMSE = RMSE(test_small$rating, y_hat_final_reco)))

# Show the RMSE improvement  
result_small


# Xgboost------
# genres is same as edx data set
genre_name 

# Copy the modify data sets
train_small2 <- train_small
test_small2 <- test_small

# To use Xgboost, modify genres such like one-hot encoding(but not same)
for (n in genre_name){
  train_small2 <- train_small2 %>% 
    mutate(!!n := if_else(str_detect(genres,n),1,0))
}
for (n in genre_name){
  test_small2 <- test_small2 %>% 
    mutate(!!n := if_else(str_detect(genres,n),1,0))
}


train_small2 %>% select(-c(title, genres)) %>% 
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> train_small2

test_small2 %>% select(-c(title, genres)) %>%
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> test_small2

# Note: it took 45 minutes!!
# Xgboost Linear model
set.seed(1, sample.kind = "Rounding")
system.time(
  modelXgboostLinear <- train(
    rating ~ ., 
    data = train_small2,
    method = "xgbLinear",
    preProcess = c('center', 'scale'), 
    trControl = trainControl(method = "cv"),
    tuneLength = 4)
)

print(modelXgboostLinear)
ggplot(modelXgboostLinear)

# Note: it took 30 minutes!!
# Xgboost Tree model
set.seed(1, sample.kind = "Rounding")
system.time(
  modelXgboostTree <- train(
    rating ~ ., 
    data = train_small2,
    method = "xgbTree",
    preProcess = c('center', 'scale'), 
    trControl = trainControl(method = "cv"),
    tuneLength = 4)
)

print(modelXgboostTree)

# Calculate the prediction
y_hat_final_xgbL <-  predict(modelXgboostLinear, test_small2)

# Update the result_small table
result_small <- bind_rows(result_small, 
                          data.frame(Method = "XGboostLinear", 
                                     RMSE = RMSE(test_small$rating, y_hat_final_xgbL)))

# Show the RMSE improvement  
result_small %>% kable()


# Calculate the prediction
y_hat_final_xgbT <-  predict(modelXgboostTree, test_small2)

# Update the result_small table
result_small <- bind_rows(result_small, 
                          data.frame(Method = "XGboostTree", 
                                     RMSE = RMSE(test_small$rating, y_hat_final_xgbT)))

# Show the RMSE improvement  
result_small %>% kable()



# To make a model for new user / new movie (ONLY genre selection)
train_small2 %>% select(-c(movieId, year, userId, timestamp)) %>% 
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> train_small3

test_small2 %>% select(-c(movieId, year, userId, timestamp)) %>%
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> test_small3



# Note: it took 40 minutes!!
# Xgboost Linear model for new user/movie
set.seed(1, sample.kind = "Rounding")
system.time(
  modelXgboostLinear_new <- train(
    rating ~ ., 
    data = train_small3,
    method = "xgbLinear",
    preProcess = c('center', 'scale'), 
    trControl = trainControl(method = "cv"),
    tuneLength = 4)
)

print(modelXgboostLinear_new)
ggplot(modelXgboostLinear_new)



# Calculate the prediction
y_hat_final_xgbL_new <-  predict(modelXgboostLinear_new, test_small3)

# Update the result_small table
result_small <- bind_rows(result_small, 
                          data.frame(Method = "XGboostLinear for New User/Movie", 
                                     RMSE = RMSE(test_small$rating, y_hat_final_xgbL_new)))

# Show the RMSE improvement  
result_small %>% kable()




# Again, to make a model for new user (KEEP the movieId)
train_small2 %>% select(-c(year, userId, timestamp)) %>% 
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> train_small4

test_small2 %>% select(-c(year, userId, timestamp)) %>%
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> test_small4



# Note: it took 35 minutes!!
# Xgboost Linear model for new user/movie
set.seed(1, sample.kind = "Rounding")
system.time(
  modelXgboostLinear_newUser <- train(
    rating ~ ., 
    data = train_small4,
    method = "xgbLinear",
    preProcess = c('center', 'scale'), 
    trControl = trainControl(method = "cv"),
    tuneLength = 4)
)

print(modelXgboostLinear_newUser)
ggplot(modelXgboostLinear_newUser)



# Calculate the prediction
y_hat_final_xgbL_newUser <-  predict(modelXgboostLinear_newUser, test_small4)

# Update the result_small table
result_small <- bind_rows(result_small, 
                          data.frame(Method = "XGboostLinear for New User", 
                                     RMSE = RMSE(test_small$rating, y_hat_final_xgbL_newUser)))

# Show the RMSE improvement  
result_small %>% kable()






# Once more, to make a model for new movie (KEEP the userId)
train_small2 %>% select(-c(year, movieId, timestamp)) %>% 
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> train_small5

test_small2 %>% select(-c(year, movieId, timestamp)) %>%
  mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .)) -> test_small5



# Note: it took 35 minutes!!
# Xgboost Linear model for new user/movie
set.seed(1, sample.kind = "Rounding")
system.time(
  modelXgboostLinear_newMovie <- train(
    rating ~ ., 
    data = train_small5,
    method = "xgbLinear",
    preProcess = c('center', 'scale'), 
    trControl = trainControl(method = "cv"),
    tuneLength = 4)
)

print(modelXgboostLinear_newMovie)
ggplot(modelXgboostLinear_newMovie)



# Calculate the prediction
y_hat_final_xgbL_newMovie <-  predict(modelXgboostLinear_newMovie, test_small5)

# Update the result_small table
result_small <- bind_rows(result_small, 
                          data.frame(Method = "XGboostLinear for New Movie", 
                                     RMSE = RMSE(test_small$rating, y_hat_final_xgbL_newMovie)))

# Show the RMSE improvement  
result_small %>% kable()
