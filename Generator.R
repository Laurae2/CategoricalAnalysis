library(rpart)
library(data.table)
library(R.utils)
library(microbenchmark)
library(doParallel)
library(foreach)

setwd("D:/CategoricalAnalysis")

# Grid for analysis
gridlist <- 2^(3:13)
gridlisting <- list()
for (i in 1:length(gridlist)) {
  for (j in 1:2) {
    gridlisting[[2 * (i - 1) + j]] <- c(gridlist[i], gridlist[i] / (2 ^ (3 - j)))
  }
}

# Prepare analysis of grid
runs <- 250 # how many runs per analysis? multiple of 5
times <- c(25, 25, 1, 10) # categorical runs, numerical runs, one-hot encoding runs, binary encoding runs per run
big_data <- data.table(Categories = numeric(length(gridlisting) * runs),
                       Correctness = numeric(length(gridlisting) * runs ),
                       Run = numeric(length(gridlisting) * runs),
                       L1_ExtremaSum = numeric(length(gridlisting) * runs),
                       L2_ExtremaSum = numeric(length(gridlisting) * runs),
                       Categorical_Acc = numeric(length(gridlisting) * runs),
                       Numerical_Acc = numeric(length(gridlisting) * runs),
                       OneHotEncoding_Acc = numeric(length(gridlisting) * runs),
                       BinaryEncoding_Acc = numeric(length(gridlisting) * runs),
                       Categorical_Time = numeric(length(gridlisting) * runs),
                       Numerical_Time = numeric(length(gridlisting) * runs),
                       OneHotEncoding_Time = numeric(length(gridlisting) * runs),
                       BinaryEncoding_Time = numeric(length(gridlisting) * runs))
replacor <- 0 # just row ID for loop

pbar <- winProgressBar(title = "Decision Tree Experiment",
                       label = paste0("Progress done: 0% (0 / ", nrow(big_data), ") - 0s done, 0s ETA"),
                       min = 0,
                       max = nrow(big_data),
                       initial = 0,
                       width = 600)

mcl <- makeCluster(6)
invisible(clusterEvalQ(mcl, library("R.utils")))
invisible(clusterEvalQ(mcl, library("data.table")))
invisible(clusterEvalQ(mcl, library("rpart")))
invisible(clusterEvalQ(mcl, library("microbenchmark")))
registerDoParallel(cl = mcl)
clusterExport(mcl, c("times"))

timed <- System$currentTimeMillis()

# Loop for each element of grid
for (j in 1:length(gridlisting)) {
  
  categories <- gridlisting[[j]][1] # must be larger than correctness
  correctness <- gridlisting[[j]][2] # must be pair
  values <- 1 # number of samples multiplication, not needing more than 1
  
  clusterExport(mcl, c("categories", "correctness", "values"))
  
  for (i in 1:5) {
    
    clusterExport(mcl, "replacor")
    
    whatever <- foreach(k = ((i - 1) * (runs / 5) + 1):(i * (runs / 5)), .combine = "rbind", .inorder = TRUE, .noexport = c("categories", "correctness", "values", "times", "replacor", "new_data", "my_data", "my_labels", "set_to_1", "real_labels", "ordered_labels", "out_vector")) %dopar% {
      
      # Adjust ID
      out_vector <- numeric(13)
      out_vector[1] <- categories
      out_vector[2] <- correctness
      out_vector[3] <- k
      
      # Generate data
      gc(verbose = FALSE)
      set.seed(replacor + k)
      my_data <- rep(sample(1:categories, size = categories, replace = FALSE) - 1, values)
      my_labels <- sample(1:categories, size = correctness, replace = FALSE) - 1
      set_to_1 <- which(my_data %in% my_labels)
      real_labels <- numeric(categories * values)
      real_labels[set_to_1] <- 1
      ordered_labels <- my_labels[order(my_labels)]
      out_vector[4] <- sum(ordered_labels[correctness:(correctness / 2 + 1)] - ordered_labels[1:(correctness / 2)]) / (correctness / 2)
      out_vector[5] <- sqrt(sum((ordered_labels[correctness:(correctness / 2 + 1)] - ordered_labels[1:(correctness / 2)]) ^ 2) / (correctness / 2))
      
      # For categorical
      gc(verbose = FALSE)
      new_data <- data.table(feature = my_data, label = real_labels)
      new_data[["label"]] <- as.factor(new_data[["label"]])
      new_data[["feature"]] <- as.factor(new_data[["feature"]])
      out_vector[10] <- median(microbenchmark({my_model <- rpart(label ~ .,
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
                                                                                        maxdepth = 30))},
                                             times = times[1])$time) / 1000000000
      new_labels <- predict(my_model, new_data, type = "vector") - 1
      out_vector[6] <- sum(new_labels == real_labels) / (categories * values)
      
      # For numeric
      gc(verbose = FALSE)
      new_data <- data.table(feature = my_data, label = real_labels)
      new_data[["label"]] <- as.factor(new_data[["label"]])
      out_vector[11] <- median(microbenchmark({my_model <- rpart(label ~ .,
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
                                                                                        maxdepth = 30))},
                                             times = times[2])$time) / 1000000000
      new_labels <- predict(my_model, new_data, type = "vector") - 1
      out_vector[7] <- sum(new_labels == real_labels) / (categories * values)
      
      # For one-hot encoding
      gc(verbose = FALSE)
      new_data <- data.table(feature = my_data, label = real_labels)
      new_data[["feature"]] <- as.factor(new_data[["feature"]])
      new_data <- data.table(model.matrix(~ feature + label + 0, new_data))
      new_data[["label"]] <- as.factor(new_data[["label"]])
      out_vector[12] <- median(microbenchmark({my_model <- rpart(label ~ .,
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
                                                                                        maxdepth = 30))},
                                             times = times[3])$time) / 1000000000
      new_labels <- predict(my_model, new_data, type = "vector") - 1
      out_vector[8] <- sum(new_labels == real_labels) / (categories * values)
      
      # For binary encoding
      gc(verbose = FALSE)
      new_data <- data.table(cbind(label = real_labels, t(array(as.integer(intToBits(my_data)), dim = c(32, (categories * values))))))
      colnames(new_data)[2:ncol(new_data)] <- paste0("feature", 1:(ncol(new_data) - 1))
      new_data[["label"]] <- as.factor(new_data[["label"]])
      out_vector[13] <- median(microbenchmark({my_model <- rpart(label ~ .,
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
                                                                                        maxdepth = 30))},
                                             times = times[4])$time) / 1000000000
      new_labels <- predict(my_model, new_data, type = "vector") - 1
      out_vector[9] <- sum(new_labels == real_labels) / (categories * values)
      
      return(out_vector)
      
    }
    
    colnames(whatever) <- colnames(big_data)
    big_data[(replacor + 1):(replacor + runs / 5), 1:13] <- data.table(whatever)
    
    replacor <- replacor + (runs / 5)
    
    setWinProgressBar(pbar,
                      value = replacor,
                      label = paste0("Progress done: ", sprintf("%05.02f", 100 * replacor / nrow(big_data)), "% (", replacor, " / ", nrow(big_data), ") - ", sprintf("%.03f", (System$currentTimeMillis() - timed) / 1000), "s, ", sprintf("%.03f", (System$currentTimeMillis() - timed) / 1000 * (nrow(big_data) / replacor)), "s ETA"))
    
  }
  
  fwrite(big_data, "big_data.csv")
  
}

registerDoSEQ()
stopCluster(mcl)
closeAllConnections()

close(pbar)











# # OLD SYSTEM
# 
# # Loop for each element of grid
# for (j in 1:length(gridlisting)) {
#   
#   categories <- gridlisting[[j]][1] # must be larger than correctness
#   correctness <- gridlisting[[j]][2] # must be pair
#   values <- 1 # number of samples multiplication, not needing more than 1
#   
#   for (i in 1:runs) {
#     
#     # Adjust ID
#     replacor <- replacor + 1
#     big_data[["Categories"]][replacor] <- categories
#     big_data[["Correctness"]][replacor] <- correctness
#     big_data[["Run"]][replacor] <- i
#     
#     
#     # Generate data
#     gc(verbose = FALSE)
#     set.seed(replacor)
#     my_data <- rep(sample(1:categories, size = categories, replace = FALSE) - 1, values)
#     my_labels <- sample(1:categories, size = correctness, replace = FALSE) - 1
#     set_to_1 <- which(my_data %in% my_labels)
#     real_labels <- numeric(categories * values)
#     real_labels[set_to_1] <- 1
#     ordered_labels <- my_labels[order(my_labels)]
#     big_data[["L1_ExtremaSum"]][replacor] <- sum(ordered_labels[correctness:(correctness / 2 + 1)] - ordered_labels[1:(correctness / 2)]) / (correctness / 2)
#     big_data[["L2_ExtremaSum"]][replacor] <- sqrt(sum((ordered_labels[correctness:(correctness / 2 + 1)] - ordered_labels[1:(correctness / 2)]) ^ 2) / (correctness / 2))
#     
#     # For categorical
#     gc(verbose = FALSE)
#     new_data <- data.table(feature = my_data, label = real_labels)
#     new_data[["label"]] <- as.factor(new_data[["label"]])
#     new_data[["feature"]] <- as.factor(new_data[["feature"]])
#     big_data[["Categorical_Time"]][replacor] <- median(microbenchmark({my_model <- rpart(label ~ .,
#                                                                                          data = new_data,
#                                                                                          method = "class",
#                                                                                          parms = list(split = "information"),
#                                                                                          control = rpart.control(minsplit = 1,
#                                                                                                                  minbucket = 1,
#                                                                                                                  cp = 1e-15,
#                                                                                                                  maxcompete = 1,
#                                                                                                                  maxsurrogate = 1,
#                                                                                                                  usesurrogate = 0,
#                                                                                                                  xval = 1,
#                                                                                                                  surrogatestyle = 1,
#                                                                                                                  maxdepth = 30))},
#                                                                       times = times[1])$time) / 1000000000
#     new_labels <- predict(my_model, new_data, type = "vector") - 1
#     big_data[["Categorical_Acc"]][replacor] <- sum(new_labels == real_labels) / (categories * values)
#     
#     # For numeric
#     gc(verbose = FALSE)
#     new_data <- data.table(feature = my_data, label = real_labels)
#     new_data[["label"]] <- as.factor(new_data[["label"]])
#     big_data[["Numerical_Time"]][replacor] <- median(microbenchmark({my_model <- rpart(label ~ .,
#                                                                                      data = new_data,
#                                                                                      method = "class",
#                                                                                      parms = list(split = "information"),
#                                                                                      control = rpart.control(minsplit = 1,
#                                                                                                              minbucket = 1,
#                                                                                                              cp = 1e-15,
#                                                                                                              maxcompete = 1,
#                                                                                                              maxsurrogate = 1,
#                                                                                                              usesurrogate = 0,
#                                                                                                              xval = 1,
#                                                                                                              surrogatestyle = 1,
#                                                                                                              maxdepth = 30))},
#                                                                   times = times[2])$time) / 1000000000
#     new_labels <- predict(my_model, new_data, type = "vector") - 1
#     big_data[["Numerical_Acc"]][replacor] <- sum(new_labels == real_labels) / (categories * values)
#     
#     # For one-hot encoding
#     gc(verbose = FALSE)
#     new_data <- data.table(feature = my_data, label = real_labels)
#     new_data[["feature"]] <- as.factor(new_data[["feature"]])
#     new_data <- data.table(model.matrix(~ feature + label + 0, new_data))
#     new_data[["label"]] <- as.factor(new_data[["label"]])
#     big_data[["OneHotEncoding_Time"]][replacor] <- median(microbenchmark({my_model <- rpart(label ~ .,
#                                                                                             data = new_data,
#                                                                                             method = "class",
#                                                                                             parms = list(split = "information"),
#                                                                                             control = rpart.control(minsplit = 1,
#                                                                                                                     minbucket = 1,
#                                                                                                                     cp = 1e-15,
#                                                                                                                     maxcompete = 1,
#                                                                                                                     maxsurrogate = 1,
#                                                                                                                     usesurrogate = 0,
#                                                                                                                     xval = 1,
#                                                                                                                     surrogatestyle = 1,
#                                                                                                                     maxdepth = 30))},
#                                                                          times = times[3])$time) / 1000000000
#     new_labels <- predict(my_model, new_data, type = "vector") - 1
#     big_data[["OneHotEncoding_Acc"]][replacor] <- sum(new_labels == real_labels) / (categories * values)
#     
#     # For binary encoding
#     gc(verbose = FALSE)
#     new_data <- data.table(cbind(label = real_labels, t(array(as.integer(intToBits(my_data)), dim = c(32, (categories * values))))))
#     colnames(new_data)[2:ncol(new_data)] <- paste0("feature", 1:(ncol(new_data) - 1))
#     new_data[["label"]] <- as.factor(new_data[["label"]])
#     big_data[["BinaryEncoding_Time"]][replacor] <- median(microbenchmark({my_model <- rpart(label ~ .,
#                                                                                             data = new_data,
#                                                                                             method = "class",
#                                                                                             parms = list(split = "information"),
#                                                                                             control = rpart.control(minsplit = 1,
#                                                                                                                     minbucket = 1,
#                                                                                                                     cp = 1e-15,
#                                                                                                                     maxcompete = 1,
#                                                                                                                     maxsurrogate = 1,
#                                                                                                                     usesurrogate = 0,
#                                                                                                                     xval = 1,
#                                                                                                                     surrogatestyle = 1,
#                                                                                                                     maxdepth = 30))},
#                                                                          times = times[4])$time) / 1000000000
#     new_labels <- predict(my_model, new_data, type = "vector") - 1
#     big_data[["BinaryEncoding_Acc"]][replacor] <- sum(new_labels == real_labels) / (categories * values)
#     
#     setWinProgressBar(pbar,
#                       value = replacor,
#                       label = paste0("Progress done: ", sprintf("%05.02f", 100 * replacor / nrow(big_data)), "% (", replacor, " / ", nrow(big_data), ") - ", sprintf("%.03f", (System$currentTimeMillis() - timed) / 1000), "s, ", sprintf("%.03f", (System$currentTimeMillis() - timed) / 1000 * (nrow(big_data) / replacor)), "s ETA"))
#     
#   }
#   
#   fwrite(big_data, "big_data.csv")
#   
# }
# 
# close(pbar)
