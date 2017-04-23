library(rpart)
library(data.table)

categories <- 1024 # must be larger than correctness
correctness <- 256 # must be pair
values <- 1 # must be greater than 1

set.seed(11111)
my_data <- sample(1:categories, size = categories * values, replace = TRUE) - 1
my_labels <- sample(1:categories, size = correctness, replace = FALSE) - 1
set_to_1 <- which(my_data %in% my_labels)
real_labels <- numeric(categories * values)
real_labels[set_to_1] <- 1
ordered_labels <- my_labels[order(my_labels)]
l1_diff_sum <- sum(ordered_labels[correctness:(correctness / 2 + 1)] - ordered_labels[1:(correctness / 2)]) / (correctness / 2)
l2_diff_sum <- sqrt(sum((ordered_labels[correctness:(correctness / 2 + 1)] - ordered_labels[1:(correctness / 2)]) ^ 2) / (correctness / 2))

# For categorical
new_data <- data.table(feature = my_data, label = real_labels)
new_data[["label"]] <- as.factor(new_data[["label"]])
new_data[["feature"]] <- as.factor(new_data[["feature"]])
my_model <- rpart(label ~ .,
                  data = new_data,
                  method = "class",
                  parms = list(split = "information"),
                  control = rpart.control(minsplit = 1,
                                          minbucket = 1,
                                          cp = 1e-15,
                                          maxcompete = 1,
                                          maxsurrogate = 1,
                                          usesurrogate = 0,
                                          xval = 1,
                                          surrogatestyle = 1,
                                          maxdepth = 30))
new_labels <- predict(my_model, new_data, type = "vector") - 1
sum(new_labels == real_labels) / (categories * values)
plot(my_model)
title(paste0("Categorical Encoding - Accuracy: ", sprintf("%05.02f", 100 * sum(new_labels == real_labels) / (categories * values)), "%"))

# For numeric
new_data <- data.table(feature = my_data, label = real_labels)
new_data[["label"]] <- as.factor(new_data[["label"]])
my_model <- rpart(label ~ .,
                  data = new_data,
                  method = "class",
                  parms = list(split = "information"),
                  control = rpart.control(minsplit = 1,
                                          minbucket = 1,
                                          cp = 1e-15,
                                          maxcompete = 1,
                                          maxsurrogate = 1,
                                          usesurrogate = 0,
                                          xval = 1,
                                          surrogatestyle = 1,
                                          maxdepth = 30))
new_labels <- predict(my_model, new_data, type = "vector") - 1
sum(new_labels == real_labels) / (categories * values)
plot(my_model)
title(paste0("Numeric Encoding - Accuracy: ", sprintf("%05.02f", 100 * sum(new_labels == real_labels) / (categories * values)), "%"))

# For one-hot encoding
new_data <- data.table(feature = my_data, label = real_labels)
new_data[["feature"]] <- as.factor(new_data[["feature"]])
new_data <- data.table(model.matrix(~ feature + label + 0, new_data))
new_data[["label"]] <- as.factor(new_data[["label"]])
my_model <- rpart(label ~ .,
                  data = new_data,
                  method = "class",
                  parms = list(split = "information"),
                  control = rpart.control(minsplit = 1,
                                          minbucket = 1,
                                          cp = 1e-15,
                                          maxcompete = 1,
                                          maxsurrogate = 1,
                                          usesurrogate = 0,
                                          xval = 1,
                                          surrogatestyle = 1,
                                          maxdepth = 30))
new_labels <- predict(my_model, new_data, type = "vector") - 1
sum(new_labels == real_labels) / (categories * values)
plot(my_model)
title(paste0("One-Hot Encoding - Accuracy: ", sprintf("%05.02f", 100 * sum(new_labels == real_labels) / (categories * values)), "%"))

# For binary encoding
new_data <- data.table(cbind(label = real_labels, t(array(as.integer(intToBits(my_data)), dim = c(32, (categories * values))))))
colnames(new_data)[2:ncol(new_data)] <- paste0("feature", 1:(ncol(new_data) - 1))
new_data[["label"]] <- as.factor(new_data[["label"]])
my_model <- rpart(label ~ .,
                  data = new_data,
                  method = "class",
                  parms = list(split = "information"),
                  control = rpart.control(minsplit = 1,
                                          minbucket = 1,
                                          cp = 1e-15,
                                          maxcompete = 1,
                                          maxsurrogate = 1,
                                          usesurrogate = 0,
                                          xval = 1,
                                          surrogatestyle = 1,
                                          maxdepth = 30))
new_labels <- predict(my_model, new_data, type = "vector") - 1
sum(new_labels == real_labels) / (categories * values)
plot(my_model)
title(paste0("Binary Encoding - Accuracy: ", sprintf("%05.02f", 100 * sum(new_labels == real_labels) / (categories * values)), "%"))
