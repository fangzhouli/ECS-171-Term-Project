library(data.table)

set.seed(1234)

# Run logistic regression on 'features.csv'
# Input:
#   - path: str,
#       path to the dataset.
#   - n_train: int, default 40113
#       Number of rows from the beginning to be used for training. Default is
#       40113 because 2018 matches start from 40114.
#   - random: boolean, default FALSE
#       Flag to indicate program to randomly choose training samples.
run <- function(path, n_train, random)
{
  options(warn=-1)
  cat(paste0("Running ", path, '\n'))
  # Remove index column
  dt <- fread(path, drop=1)

  ns <- colnames(dt)
  ns[length(ns)] <- 'win'
  colnames(dt) <- ns

  if(random)
    trainIdx <- sample(nrow(dt), n_train)
  else
    trainIdx <- 1:n_train

  # Split dataset
  trainSet <- dt[trainIdx,]
  testSet <- dt[-trainIdx,]

  # Weights Training
  lmout <- glm(win ~ ., data=trainSet, family=binomial)

  # Cross Validation
  y_train = round(predict(lmout, trainSet[, -ncol(trainSet), with=F],
                          type='response'))
  y_test = round(predict(lmout, testSet[, -ncol(testSet), with=F],
                         type='response'))
  # Accruacy Computation
  acc_train <- mean(y_train==trainSet[,ncol(trainSet),with=F])
  acc_test <- mean(y_test==testSet[,ncol(testSet),with=F])

  cat(paste0("Training Accuracy:\t", round(acc_train, 4), '\n'))
  cat(paste0("Testing Accuracy:\t", round(acc_test, 4), '\n'))
}

run('./data_processing/output/post_game_team_diff.csv',
    n_train=27524, random=T)
run('./pre_game_teams.csv', n_train=40113, random=F)
