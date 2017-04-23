library(data.table)
library(ggplot2)
setwd("D:/CategoricalAnalysis")

data <- fread("big_data.csv", colClasses = c("integer", "integer", "integer",
                                             "numeric", "numeric",
                                             "numeric", "numeric", "numeric", "numeric",
                                             "numeric", "numeric", "numeric", "numeric"))

data$PercentageCor <- data$Correctness / data$Categories

new_data <- rbindlist(list(data[, c(1:5, 6, 10, 14), with = FALSE],
                           data[, c(1:5, 7, 11, 14), with = FALSE],
                           data[, c(1:5, 8, 12, 14), with = FALSE],
                           data[, c(1:5, 9, 13, 14), with = FALSE]))
colnames(new_data) <- c("Categories", "Correctness", "Run", "L1_ExtremaSum", "L2_ExtremaSum", "Acc", "Time", "PercentageCor")
new_data$Model <- factor(c(rep("Categorical Encoding", nrow(data)), rep("Numeric Encoding", nrow(data)), rep("One-Hot Encoding", nrow(data)), rep("Binary Encoding", nrow(data))), levels = c("Categorical Encoding", "Numeric Encoding", "One-Hot Encoding", "Binary Encoding"))
new_data$Categories <- as.factor(new_data$Categories)
new_data$Correctness <- as.factor(new_data$Correctness)
new_data <- new_data[, Mean_Acc := mean(Acc), by = list(Categories, Correctness, Model)]
new_data <- new_data[, Mean_Time := mean(Time), by = list(Categories, Correctness, Model)]
new_data <- new_data[, Sd_Acc := sd(Acc), by = list(Categories, Correctness, Model)]
new_data <- new_data[, Sd_Time := sd(Time), by = list(Categories, Correctness, Model)]

# MAKE HANDS EDIT NOW AND RUN INTERACTIVELY IN RStudio
# ANALYZE YOURSELF FROM HERE

to_analyze <- new_data[Categories %in% c(8, 16, 32, 64, 128, 256)]
to_analyze <- new_data[Categories %in% c(512, 1024, 2048, 4096, 8192)]
to_analyze <- new_data[Model %in% c("Categorical Encoding", "Numeric Encoding", "Binary Encoding")]
to_analyze <- new_data

mean(to_analyze[Model == "Categorical Encoding", Acc])
mean(to_analyze[Model == "Numeric Encoding", Acc])
mean(to_analyze[Model == "One-Hot Encoding", Acc])
mean(to_analyze[Model == "Binary Encoding", Acc])

ggplot(data = to_analyze, aes(x = Categories, y = Acc, fill = Model)) + geom_boxplot() + stat_summary(fun.y = mean, geom = "line", aes(color = Model, group = Model)) + labs(title = "Accuracy of Decision Tree modeled by Encoding", x = "Number of Cardinalities", y = "Accuracy") + theme_bw()

ggplot(data = to_analyze, aes(x = Categories, y = Acc, fill = Model)) + geom_boxplot() + geom_line(aes(x = Categories, y = Mean_Acc, group = Model, color = Model)) + facet_grid(PercentageCor ~ ., labeller = labeller(PercentageCor = as_labeller(c(`0.25` = "Positive: 25%", `0.5` = "Positive: 50%")))) + labs(title = "Accuracy of Decision Tree modeled by Encoding (Facetted)", x = "Number of Cardinalities", y = "Accuracy") + theme_bw()

ggplot(data = to_analyze, aes(x = Categories, y = Sd_Acc, fill = Model)) + geom_line(aes(x = Categories, y = Sd_Acc, group = Model, color = Model)) + facet_grid(PercentageCor ~ ., labeller = labeller(PercentageCor = as_labeller(c(`0.25` = "Positive: 25%", `0.5` = "Positive: 50%")))) + labs(title = "Consistency of Decision Tree modeled by Encoding (Facetted)", x = "Number of Cardinalities", y = "Accuracy Standard Deviation") + theme_bw()

ggplot(data = to_analyze, aes(x = Categories, y = Time, fill = Model)) + geom_boxplot() + stat_summary(fun.y = mean, geom = "line", aes(color = Model, group = Model)) + labs(title = "Computation Time of Decision Tree modeled by Encoding", x = "Number of Cardinalities", y = "Speed (seconds)") + theme_bw()

ggplot(data = to_analyze, aes(x = Categories, y = Time, fill = Model)) + geom_boxplot() + geom_line(aes(x = Categories, y = Mean_Time, group = Model, color = Model)) + facet_grid(PercentageCor ~ ., labeller = labeller(PercentageCor = as_labeller(c(`0.25` = "Positive: 25%", `0.5` = "Positive: 50%")))) + labs(title = "Computation Time of Decision Tree modeled by Encoding (Facetted)", x = "Number of Cardinalities", y = "Speed (seconds)") + theme_bw()
