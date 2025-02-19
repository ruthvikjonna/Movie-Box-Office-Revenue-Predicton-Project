# Load Libraries
library(tidyverse)
library(lubridate)
library(caret)
library(rpart)
library(rattle)
library(rsample)
library(modelr)

# Clears memory
rm(list = ls())
# Clears console
cat("\014")
# Remove scientific notation
options(scipen = 999)

# Load Dataset
movies = read.csv("https://raw.githubusercontent.com/ruthvikjonna/Movie-Box-Office-Revenue-Predicton-Project/main/movies_metadata.csv")

# Data Cleaning
movies = na.omit(movies)

movies = movies %>%
  filter(budget > 0, status == "Released") %>%
  select(-title, -adult, -belongs_to_collection, -genres, -homepage, -id, 
         -imdb_id, -original_language, -original_title, -overview, 
         -poster_path, -production_companies, -production_countries, 
         -spoken_languages, -status, -tagline, -video)

movies$release_date <- year(as.Date(movies$release_date))

movies = movies %>%
  mutate(years_since_1900 = release_date - 1900,
         popularity_score = popularity)

movies = movies %>%
  mutate(revenue_mil = revenue / 1000000,
         budget = as.numeric(budget),
         budget_mil = budget / 1000000,
         popularity_score = as.numeric(popularity_score))
         

movies = movies %>%
  select(revenue_mil, budget_mil, years_since_1900, runtime, popularity_score, 
         vote_average, vote_count)

movies = na.omit(movies)

# Observe Data
names(movies)
head(movies)
summary(movies$revenue)

# General Linear Regression
lm_general = lm(revenue_mil ~ budget_mil + years_since_1900 + runtime + 
                popularity_score + vote_average + vote_count, data = movies)
summary(lm_general)

# Revenue vs. Budget
ggplot(data = movies, aes(x = budget_mil, y = revenue_mil)) +
  geom_point(size = 1, color = "orange") +
  geom_smooth(color = "blue", se = FALSE, method = "lm") +
  xlab("Movie Budget (in millions)") + 
  ylab("Box Office Revenue (in millions)") +
  theme_minimal()

# Revenue vs. Vote Average
ggplot(data = movies, aes(x = vote_average, y = revenue_mil)) +
  geom_point(size = 1, color = "red") +
  xlab("Movie Budget (in millions)") + 
  ylab("Box Office Revenue (in millions)") +
  theme_minimal()

# Now, our goal is to predict the box office revenue for a movie based on 
# various factors such as movie budget, movie release year (from 1900), 
# movie runtime, movie popularity score, voting average, and voting count. 

# Splitting Data
set.seed(123)

train_id = sample(1:nrow(movies), 0.7*nrow(movies))

train.data = movies %>% slice(train_id)
test.data = movies %>% slice(-train_id)

# Simple Linear Regression
lm_simple = lm(revenue_mil ~ budget_mil + years_since_1900 + runtime + 
              popularity_score + vote_average + vote_count, data = train.data)
summary(lm_simple)
rmse(lm_simple, test.data)

# Simple Linear Regression (10 Fold Cross-Validation)
set.seed(123)

lm_simple_cv = train(revenue_mil ~ ., method = "lm", data = movies,
              trControl = trainControl(method = "cv", number = 10))
summary(lm_simple_cv)
rmse(lm_simple_cv, test.data)

# Forward Stepwise Selection
set.seed(123)

num_var = length(lm_simple_cv$coefnames)

lm_fwd = train(revenue_mil ~ ., data = train.data,
               trControl = trainControl(method = "cv", number = 10),
               method = "leapForward",
               tuneGrid = data.frame(nvmax = 1:num_var))

plot(lm_fwd)
lm_fwd$results
lm_fwd$bestTune
rmse(lm_fwd, test.data)

# Backward Stepwise Selection
set.seed(123)

num_var = length(lm_simple_cv$coefnames)

lm_bwd = train(revenue_mil ~ ., data = train.data,
               trControl = trainControl(method = "cv", number = 10),
               method = "leapBackward",
               tuneGrid = data.frame(nvmax = 1:num_var))

plot(lm_bwd)
lm_bwd$results
lm_bwd$bestTune
rmse(lm_bwd, test.data)

# Ridge Regression
set.seed(123)

ridge = train(revenue_mil ~ ., data = train.data,
              trControl = trainControl(method = "cv", number = 10),
              method = "glmnet",
              preProcess = "scale",
              tuneGrid = expand.grid(alpha = 0,
                                     lambda = seq(0, 0.1, length = 10)))

plot(ridge)
ridge$results
ridge$bestTune
summary(ridge$finalModel)
rmse(ridge, test.data)

# Lasso Regression
set.seed(123)

lasso = train(revenue_mil ~ ., data = train.data,
              trControl = trainControl(method = "cv", number = 10),
              method = "glmnet",
              preProcess = "scale",
              tuneGrid = expand.grid(alpha = 1,
                                     lambda = seq(0, 0.1, length = 10)))

plot(lasso)
lasso$results
lasso$bestTune
summary(lasso$finalModel)
rmse(lasso, test.data)

# Regression Tree
set.seed(123)

rt = train(revenue_mil ~ ., data = train.data, 
           trControl = trainControl(method = "cv", number = 10),
           method = "rpart",
           tuneGrid = expand.grid(cp = seq(0, 0.01, length = 50)))

plot(rt)
rt$results
rt$bestTune
summary(rt$finalModel)
rmse(rt, test.data)

# Simple Linear Regression RMSE: 69.38822
# Simple Linear Regression (10 Fold Cross-Validation) RMSE: 68.78104
# Forward Stepwise Selection RMSE: 69.38822
# Backward Stepwise Selection RMSE: 69.38822
# Ridge Regression RMSE: 69.96866
# Lasso Regression RMSE: 69.44357
# Regression Tree RMSE: 70.94926

# From conducting these various regression methods on our data, it can be seen
# that the simple linear regression with a 10 fold cross validation proves to
# be the most accurate model when predicting a movie's box office revenue 
# (considering these specific factors).

# Now, we will change our goal to classify movies by their success level (high, 
# medium, or low) based on various factors such as movie budget, movie release 
# year (from 1900), movie runtime, movie popularity score, voting average, and 
# voting count.

# Data Cleaning
movies_c = movies

# We will identify highly successful movies as movies who garner a revenue 
# total that is more than or equal to twice their budget amount. We will 
# identify movies with a medium-level of success as movies who garner a revenue 
# total that are less than twice their budget amount, but still are profitable 
# (above or break even). Finally, we will identify movies whose revenue amount
# is lower than their budget amount (unprofitable) to be low successful movies.

movies_c = movies_c %>%
  mutate(
    success_level = case_when(
      revenue_mil >= 2 * budget_mil ~ "high",
      revenue_mil >= budget_mil ~ "medium",
      TRUE ~ "low"
    )
  )

table(movies_c$success_level)

movies_c = movies_c %>%
  select(success_level, revenue_mil, budget_mil, years_since_1900, runtime, 
         popularity_score, vote_average, vote_count)

# Splitting Data
set.seed(123)

train_id = sample(1:nrow(movies_c), 0.7*nrow(movies_c))

train.data2 = movies_c %>% slice(train_id)
test.data2 = movies_c %>% slice(-train_id)

# Classification (Decision) Tree
set.seed(123)

ct = train(factor(success_level) ~ ., data = train.data2,
           method = "rpart", 
           tuneLength = 15, 
           trControl = trainControl(method = "cv", number = 10))

fancyRpartPlot(ct$finalModel, caption = "Movie Success Decision Tree")
plot(ct)
pred.values = ct %>% predict(test.data2)
mean(pred.values == test.data2$success_level)

# Bagging
set.seed(123)

bagging = train(factor(success_level) ~ ., data = train.data2,
           trControl = trainControl(method = "cv", number = 10),
           method = "treebag",
           nbagg = 10)

plot(varImp(bagging, scale = TRUE))
pred.values = bagging %>% predict(test.data2)
mean(pred.values == test.data2$success_level)

# Random Forest 
set.seed(123)

tuneGrid = expand.grid(
  mtry = 1:5, 
  splitrule = "gini", 
  min.node.size = 5 
)

set.seed(123)

rf = train(factor(success_level) ~ ., data = train.data2,
           trControl = trainControl(method = "cv", number = 10),
           method = "ranger",
           num.trees = 10,
           importance = "permutation",
           tuneGrid = tuneGrid)

plot(rf)
rf$finalModel
pred.values = rf %>% predict(test.data2)
mean(pred.values == test.data2$success_level)

# Boosting
tuneGrid = expand.grid(
  n.trees = c(10,20,30), # Number of trees that will test
  interaction.depth = c(1, 2), # Number of splits it will attempt
  shrinkage = c(0.01, 0.1), # Learning rates
  n.minobsinnode = 10 # Min observations in each node.
)

set.seed(123)

boosting = train(factor(success_level) ~ .,
              data = train.data2,
              trControl = trainControl(method = "cv", number = 10),
              method = "gbm",
              tuneGrid = tuneGrid)

plot(boosting)
boosting$finalModel
boosting$bestTune
pred.values = boosting %>% predict(test.data2)
mean(pred.values == test.data2$success_level)

# Classification (Decision) Tree Accuracy: 0.9280873
# Bagging Accuracy: 0.9740211
# Random Forest Accuracy: 0.9683735
# Boosting Accuracy: 0.8798946

# From conducting these various classification methods on our data, it can be
# seen that the bootstrap aggregating (bagging) method proves to be the most
# accurate model (97.4% accuracy) when classifying movies into having high 
# success, medium success, or low success. 
