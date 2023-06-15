library(tidyverse)
library(gridExtra)
library(ModelMetrics)
library(caret)
library(reshape2)
library(pROC)
library(effsize)
library(ScottKnottESD)

save.fig.dir <- "../output/figure/"

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse) {
  colnames(x) <- c("variable", "value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))]
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL

  if (reverse == TRUE) {
    ranking <- (max(sk_esd(df)$group) - sk_esd(df)$group) + 1
  } else {
    ranking <- sk_esd(df)$group
  }
  x$rank <- paste("Rank", ranking[as.character(x$variable)])
  return(x)
}

get.top.k.tokens <- function(df, value) {
  top.k <- df %>%
    filter(is.comment.line == "False" & file.level.ground.truth == "True" & prediction.label == "True") %>%
    group_by(test, filename) %>%
    top_n(value, token.attention.score) %>%
    select("project", "train", "test", "filename", "token", token.attention.score) %>%
    distinct()

  top.k$flag <- "topk"

  return(top.k)
}

get.top.threshold.tokens <- function(df, value) {
  top.k <- df %>%
    filter(is.comment.line == "False" & file.level.ground.truth == "True" & prediction.label == "True" & token.attention.score > value) %>%
    group_by(test, filename) %>%
    select("project", "train", "test", "filename", "token", token.attention.score) %>%
    distinct()

  top.k$flag <- "topk"

  return(top.k)
}

get.not.filtered <- function(df) {
  top.k <- df %>%
    filter(is.comment.line == "False" & file.level.ground.truth == "True" & prediction.label == "True") %>%
    group_by(test, filename) %>%
    select("project", "train", "test", "filename", "token", token.attention.score) %>%
    distinct()

  top.k$flag <- "topk"

  return(top.k)
}


prediction_dir <- "../output/prediction/DeepLineDP/within-release/"

all_files <- list.files(prediction_dir)

df_all <- NULL

for (f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
                      'camel-2.10.0', 'camel-2.11.0' ,
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2',
                      'jruby-1.5.0', 'jruby-1.7.0.preview1',
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

# Force attention score of comment line is 0
df_all[df_all$is.comment.line == "True", ]$token.attention.score <- 0

not.filtered <- get.not.filtered(df_all)
dim <- dim(not.filtered)
print("df_all dimensions: ")
print(paste("row count: ", dim[1], " col count: ", dim[2]))
print(paste("minimal attenttion score = ", min(df_all$token.attention.score)))

TOP_K = 1500
TOP_K2 = 30
THRESHOLD01 = 0.80
THRESHOLD02 = 0.10
THRESHOLD03 = 0.01
THRESHOLD04 = 1.8623564e-12
THRESHOLD05 = 0

strategy_name_1 <- paste("Top K ", TOP_K)
strategy_name_2 <- paste("Top K  ", TOP_K2)
strategy_name_3 <- paste("Threshold ", THRESHOLD01)
strategy_name_4 <- paste("Threshold ", THRESHOLD02)
strategy_name_5 <- paste("Threshold ", THRESHOLD03)
strategy_name_6 <- paste("Threshold ", THRESHOLD04)
strategy_name_7 <- paste("Threshold ", THRESHOLD05)

tmp_tops <- list(
  c(strategy_name_1, TOP_K, get.top.k.tokens),
  c(strategy_name_2, TOP_K2, get.top.k.tokens),
  c(strategy_name_3, THRESHOLD01, get.top.threshold.tokens),
  c(strategy_name_4, THRESHOLD02, get.top.threshold.tokens),
  c(strategy_name_5, THRESHOLD03, get.top.threshold.tokens),
  c(strategy_name_6, THRESHOLD04, get.top.threshold.tokens),
  c(strategy_name_7, THRESHOLD05, get.top.threshold.tokens)
)

stats_list <- list(strategy1 = 1, strategy2 = 2)

for (tokens_name in tmp_tops) {
  # Access the elements of each tuple
  name <- tokens_name[[1]]
  value <- tokens_name[[2]]
  factory <- tokens_name[[3]]

  print(paste("name: ", name))
  print(paste("value: ", value))

  tmp.top.k <- do.call(factory, args = list(df = df_all, value = value))

  dim <- dim(tmp.top.k)
  print("tokens dimensions: ")
  print(paste("row count: ", dim[1], " col count: ", dim[2]))
  print(paste("minimal attenttion score = ", min(tmp.top.k$token.attention.score)))
}