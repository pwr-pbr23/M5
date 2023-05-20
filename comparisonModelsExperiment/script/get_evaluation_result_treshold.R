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
    select("project", "train", "test", "filename", "token") %>%
    distinct()

  top.k$flag <- "topk"

  return(top.k)
}

get.top.treshold.tokens <- function(df, value) {
  top.k <- df %>%
    filter(is.comment.line == "False" & file.level.ground.truth == "True" & prediction.label == "True" & token.attention.score > value) %>%
    group_by(test, filename) %>%
    select("project", "train", "test", "filename", "token") %>%
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

## prepare data for baseline
line.ground.truth <- select(df_all, project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth)
line.ground.truth <- filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
line.ground.truth <- distinct(line.ground.truth)

get.line.metrics.result <- function(baseline.df, cur.df.file) {
  baseline.df.with.ground.truth <- merge(baseline.df, cur.df.file, by = c("filename", "line.number"))

  sorted <- baseline.df.with.ground.truth %>%
    group_by(filename) %>%
    arrange(-line.score, .by_group = TRUE) %>%
    mutate(order = row_number())

  # IFA
  IFA <- sorted %>%
    filter(line.level.ground.truth == "True") %>%
    group_by(filename) %>%
    top_n(1, -order)

  ifa.list <- IFA$order

  total_true <- sorted %>%
    group_by(filename) %>%
    summarize(total_true = sum(line.level.ground.truth == "True"))

  # Recall20%LOC
  recall20LOC <- sorted %>%
    group_by(filename) %>%
    mutate(effort = round(order / n(), digits = 2)) %>%
    filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>%
    mutate(recall20LOC = correct_pred / total_true)

  recall.list <- recall20LOC$recall20LOC
  # Effort20%Recall
  effort20Recall <- sorted %>%
    merge(total_true) %>%
    group_by(filename) %>%
    mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True") / total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2) / n())

  effort.list <- effort20Recall$effort20Recall

  result.df <- data.frame(ifa.list, recall.list, effort.list)

  return(result.df)
}

all_eval_releases <- c("activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0")
# all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
#                       'camel-2.10.0', 'camel-2.11.0' ,
#                       'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2',
#                       'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
#                       'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

# Force attention score of comment line is 0
df_all[df_all$is.comment.line == "True", ]$token.attention.score <- 0

TOP_K = 1500
TRESHOLD01 = 0.5
TRESHOLD02 = 0.85
TRESHOLD03 = 0.93
TRESHOLD04 = 0.97

strategy_name_1 <- paste("Top K ", TOP_K)
strategy_name_2 <- paste("Treshold ", TRESHOLD01)
strategy_name_3 <- paste("Treshold ", TRESHOLD02)
strategy_name_4 <- paste("Treshold ", TRESHOLD03)
strategy_name_5 <- paste("Treshold ", TRESHOLD04)

tmp_tops <- list(
  c(strategy_name_1, TOP_K, get.top.k.tokens),
  c(strategy_name_2, TRESHOLD01, get.top.treshold.tokens),
  c(strategy_name_3, TRESHOLD02, get.top.treshold.tokens),
  c(strategy_name_4, TRESHOLD03, get.top.treshold.tokens)
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

  print("got tokens")

  merged_df_all <- merge(df_all, tmp.top.k, by = c("project", "train", "test", "filename", "token"), all.x = TRUE)

  merged_df_all[is.na(merged_df_all$flag), ]$token.attention.score <- 0

  ## use top-k tokens
  sum_line_attn <- merged_df_all %>%
    filter(file.level.ground.truth == "True" & prediction.label == "True") %>%
    group_by(test, filename, is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
    summarize(attention_score = sum(token.attention.score), num_tokens = n())

  sorted <- sum_line_attn %>%
    group_by(test, filename) %>%
    arrange(-attention_score, .by_group = TRUE) %>%
    mutate(order = row_number())

  ## get result from DeepLineDP
  # calculate IFA
  IFA <- sorted %>%
    filter(line.level.ground.truth == "True") %>%
    group_by(test, filename) %>%
    top_n(1, -order)

  print(paste("calculated IFA: ", name))

  total_true <- sorted %>%
    group_by(test, filename) %>%
    summarize(total_true = sum(line.level.ground.truth == "True"))

  # calculate Recall20%LOC
  recall20LOC <- sorted %>%
    group_by(test, filename) %>%
    mutate(effort = round(order / n(), digits = 2)) %>%
    filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>%
    mutate(recall20LOC = correct_pred / total_true)

  print(paste("calculated Recall20: ", name))

  # calculate Effort20%Recall
  effort20Recall <- sorted %>%
    merge(total_true) %>%
    group_by(test, filename) %>%
    mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True") / total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2) / n())

  print(paste("calculated Effort20: ", name))

  ## prepare data for plotting
  deeplinedp.ifa <- IFA$order
  deeplinedp.recall <- recall20LOC$recall20LOC
  deeplinedp.effort <- effort20Recall$effort20Recall

  deepline.dp.line.result <- data.frame(deeplinedp.ifa, deeplinedp.recall, deeplinedp.effort)
  names(deepline.dp.line.result) <- c("IFA", "Recall20%LOC", "Effort@20%Recall")
  deepline.dp.line.result$technique <- name
  stats_list[[name]] <- deepline.dp.line.result

  print(paste("added to stats_list: ", name))
}

all.line.result <- rbind(stats_list[[strategy_name_1]], stats_list[[strategy_name_2]], stats_list[[strategy_name_3]], stats_list[[strategy_name_4]], stats_list[[strategy_name_5]])

recall.result.df <- select(all.line.result, c("technique", "Recall20%LOC"))
ifa.result.df <- select(all.line.result, c("technique", "IFA"))
effort.result.df <- select(all.line.result, c("technique", "Effort@20%Recall"))

print(recall.result.df)
print(ifa.result.df)
print(effort.result.df)

recall.result.df <- preprocess(recall.result.df, FALSE)
ifa.result.df <- preprocess(ifa.result.df, TRUE)
effort.result.df <- preprocess(effort.result.df, TRUE)

ggplot(recall.result.df, aes(x = reorder(variable, -value, FUN = median), y = value)) +
  geom_boxplot() +
  facet_grid(~rank, drop = TRUE, scales = "free", space = "free") +
  ylab("Recall@Top20%LOC") +
  xlab("")
ggsave(paste0(save.fig.dir, "file-Recall@Top20LOC.pdf"), width = 8, height = 5)

ggplot(effort.result.df, aes(x = reorder(variable, value, FUN = median), y = value)) +
  geom_boxplot() +
  facet_grid(~rank, drop = TRUE, scales = "free", space = "free") +
  ylab("Effort@Top20%Recall") +
  xlab("")
ggsave(paste0(save.fig.dir, "file-Effort@Top20Recall.pdf"), width = 8, height = 5)

ggplot(ifa.result.df, aes(x = reorder(variable, value, FUN = median), y = value)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0, 175)) +
  facet_grid(~rank, drop = TRUE, scales = "free", space = "free") +
  ylab("IFA") +
  xlab("")
ggsave(paste0(save.fig.dir, "file-IFA.pdf"), width = 8, height = 5)
