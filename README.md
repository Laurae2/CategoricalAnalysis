ORIGINAL LINK: https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931

# Visiting: Categorical Features and Encoding in Decision Trees

![image](https://cloud.githubusercontent.com/assets/9083669/25314621/fc7b273a-2847-11e7-8ae8-e1aaa7c329bf.png)

When you have categorical features and you are using decision trees, you often have a major issue: how to deal with categorical features?

Often you see the following…:
* Postponing the problem: use a machine learning model which handle categorical features, the greatest of solutions!
* Deal with the problem now: design matrix, one-hot encoding, binary encoding…

Usually, you WILL want to deal with the problem now, because if you postpone the problem, it means you already found the solution:
* You do not postpone problems because in Data Science, they accumulate quickly like hell (good luck remembering every problem encountered, then come back 1 month later without thinking about them and recite them each).
* You do not postpone problems without knowing the potential remedy afterwards (otherwise, you might have a working pipeline but no solution to solve it!).

So what is the matter? Let’s go back to the basics of decision trees and encoding, then we can test some good stuff… in three parts:
* Machine Learning Implementations: specifications differ
* Example ways of Encoding categorical features
* Benchmarking Encodings versus vanilla Categorical features

## Decision Trees and Encoding

### Machine Learning Specification

When using decision tree models and categorical features, you mostly have three types of models:
* Models handling categorical features CORRECTLY. You just throw the categorical features at the model in the appropriate format (ex: as factors in R), AND the machine learning model processes categorical features correctly as categoricals. BEST CASE because it fits your needs.
* Models handling categorical features INCORRECTLY. You just throw the categorical features at the model in the appropriate format (ex: as factors in R), BUT the machine learning model processes categorical features incorrectly by doing wizardry processing to transform them into something usable (like one-hot encoding), unless you are aware of it. WORST CASE EVER because it does not do what you expected to do.
* Models NOT handling categorical features at all. You have to preprocess manually the categorical features to have them in an appropriate format for the machine learning model (usually: numeric features). But how do you transform (aka ENCODE) them?

We will target specifically the third type of model, because it is what we want to assess. There are many methods to encode categorical features. We are going to check three of them: numeric encoding, one-hot encoding, and binary encoding.

### Categorical Encoding Specification

Categorical Encoding refers to transforming a categorical feature into one or multiple numeric features. You can use any mathematical method or logical method you wish to transform the categorical feature, the sky is the limit for this task.

#### Numeric Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314624/073fbe1a-2848-11e7-867b-d999713eb073.png)

Are you transforming your categorical features by hand or are you doing the work with a computer?Numerical Encoding is very simple: assign an arbitrary number to each category. 

There is no rocket science for the transformation, except perhaps… how do you assign the arbitrary number? Is there a simple way?

The typical case is to let your favorite programming language do the work. For instance, you might do like this in R…

`my_data$cat_feature <- as.numeric(as.factor(my_data$cat_feature))`

Such as this:

```r
as.numeric(as.factor(c("Louise",
                       "Gabriel",
                       "Emma",
                       "Adam",
                       "Alice",
                       "Raphael",
                       "Chloe",
                       "Louis",
                       "Jeanne",
                       "Arthur")))
```

![image](https://cloud.githubusercontent.com/assets/9083669/25314657/a38aefb0-2848-11e7-91a5-71919ef115ff.png)

This works, this is not brainer, and it encodes the way it wants deterministically (check the ordering and you will see).

#### One-Hot Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314647/874875d4-2848-11e7-9597-014ccc9b2660.png)

One-Hot Encoding is just a design matrix with the first factor kept. A design matrix removes the first factor to avoid the matrix inversion problem in linear regressions.

Ever heard about One-Hot Encoding and its magic? Here you have it: this is a design matrix where you keep the first factor instead of removing it (how simple!).

To put it clear, just check the picture as it talks for itself better than 1,000 words.

In addition to thinking about what One-Hot Encoding does, you will notice something very quickly:
* You have as many columns as you have cardinalities (values) in the categorical variable.
* You have a bunch of zeroes and only few 1s! (one 1 per new feature)

Therefore, you have to choose between two representations of One-Hot Encoding:
* Dense Representation: 0s are stored in memory, which ballons the RAM usage a LOT if you have many cardinalities. But at least, the support for such representation is typically… worldwide.
* Sparse Repsentation: 0s are not stored in memory, which makes RAM efficiency a LOT better even if you have millions of cardinalities. However, good luck finding support for sparse matrices for machine learning, because it is not widespread (think: xgboost, LightGBM, etc.).

Again, you usually let your favorite programming language doing the work. Do not loop through each categorical value and assign a column, because this is NOT an efficient at all. It is not difficult, right?
Example in R, “one line”!:

```r
model.matrix(~ cat + 0,
             data = data.frame(
               cat = as.factor(c("Louise",
                                 "Gabriel",
                                 "Emma",
                                 "Adam",
                                 "Alice",
                                 "Raphael",
                                 "Chloe",
                                 "Louis",
                                 "Jeanne",
                                 "Arthur"))))
```

![image](https://cloud.githubusercontent.com/assets/9083669/25314649/8eedca32-2848-11e7-9131-c59a4623632c.png)

Dense One-Hot Encoding in R example. As usual, the specific order is identical to the numeric version due to as.factor choosing the order arbitrarily!If you are running out of available memory, what about working with sparse matrices? Doing it in R is no brainer in “one line”!

```r
library(Matrix)
sparse.model.matrix(~ cat + 0,
                    data = data.frame(
                      cat = as.factor(c("Louise",
                                        "Gabriel",
                                        "Emma",
                                        "Adam",
                                        "Alice",
                                        "Raphael",
                                        "Chloe",
                                        "Louis",
                                        "Jeanne",
                                        "Arthur"))))
```

Sparse One-Hot Encoding in R. There is no difference to the Dense version, except we end up with a sparse matrix (dgCMatrix: sparse column compressed matrix).

![image](https://cloud.githubusercontent.com/assets/9083669/25314650/9493b32a-2848-11e7-8e88-98c2a9158048.png)

#### Binary Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314651/9a5182ec-2848-11e7-83e8-57cf3633c3ee.png)

Power of binaries! The objective of Binary Encoding… is to use binary encoding to hash the cardinalities into binary values.
By using the power law of binary encoding, we are able to store N cardinalities using ceil(log(N+1)/log(2)) features.

It means we can store 4294967295 cardinalities using only 32 features with Binary Encoding! Isn’t it awesome to not have those 4294697295 features from One-Hot Encoding? (how are you going to learn 4 billion features in a decision tree…? you need a depth of 32 and it is not readable…)

Still as easy in (base) R, you just need to think you are limited to a specified number of bits (will you ever reach 4294967296 cardinalities? If yes, get rid of some categories because you got too many of them…):

```r
my_data <- c("Louise",
             "Gabriel",
             "Emma",
             "Adam",
             "Alice",
             "Raphael",
             "Chloe",
             "Louis",
             "Jeanne",
             "Arthur")
matrix(
  as.integer(intToBits(as.integer(as.factor(my_data)))),
  ncol = 32,
  nrow = length(my_data),
  byrow = TRUE
)[, 1:ceiling(sqrt(length(unique(my_data))))]
```

![image](https://cloud.githubusercontent.com/assets/9083669/25314662/ae9410e4-2848-11e7-9e9b-224ef7f5c8b0.png)

Binary Encoding in base R.

Ugh, the formula is a bit larger than expected. But you get the idea:

![image](https://cloud.githubusercontent.com/assets/9083669/25314667/b3e691a2-2848-11e7-9085-75b33ac29deb.png)

Three key operations to perform for binary encoding.
* Operation 1: convert my_data to factor, then to integer (“numeric”), then to numeric binary representation (as a vector of length 32 for each observation), then to integer (“numeric”).
* Operation 2: convert the “numeric” to a matrix with 32 columns and the same number of rows as the number of original observations.
* Operation 3: using the inverse binary power property (square root, aka power 0.5), remove all the unused columns (the columns with zeroes).

There are, obviously, easier ways to do this. But I am doing this example to show you can do this in base R. No need fancy package stuff.

## Benchmarking Performance of Encoding

We are going to benchmark the performance of four types of encoding:
* Categorical Encoding (raw, as is)
* Numeric Encoding
* One-Hot Encoding
* Binary Encoding

We will use rpart as the decision tree learning model, as it is also independent to random seeds.

The experimental design is the following:
* We create datasets of one categorical feature with 8 to 8,192 cardinalities (steps of power of 2).
* We use 25% or 50% of cardinalities as positive labels to assess performance of the decision tree. This means a ratio of 1:3 or 1:1.
* We run 250 times each combination of cardinalities and percentage of positive labels to get a better expected value (mean) of performance.
* To speed up the computations, we are using 6 parallel threads as One-Hot Encoding is computationally intensive.
* The rpart function is limited to a maximum depth of 30 for practical usage, and used with the following parameters:

```r
rpart(label ~ .,
      data = my_data,
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
```

Warning: remember we are doing this on a synthetic dataset with perfect rules. You may get contradictory performance on real world datasets.

### Sneak Peak: how are the decision trees looking?

For the sake of example, with 1024 categories and 25% positive labels.

* Categorical Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314671/bfe4c8de-2848-11e7-9adb-17befbd64c2b.png)

* Numeric Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314675/c4ff0136-2848-11e7-9aff-3a622df91925.png)

* One-Hot Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314678/c93cad70-2848-11e7-9ec7-bf1ea4e0388f.png)

* Binary Encoding

![image](https://cloud.githubusercontent.com/assets/9083669/25314680/d04fcac0-2848-11e7-82bd-62f3e2be36af.png)

### General Accuracy of Encodings

Without looking in depth into the accuracy, we are going to look quickly at the general accuracy of the four encodings we have.

![image](https://cloud.githubusercontent.com/assets/9083669/25314685/d9052138-2848-11e7-8594-c8c2c007d6f1.png)

On the picture, we can clearly notice a trend:
* Categorical Encoding is the clear winner, with an exact 100% Accuracy at any time.
* Numeric Encoding is doing an excellent job as long as the number of cardinalities is not too large: from 1,024 cardinalities, its accuracy falls off drastically. Its accuracy over all the tests is around 92%.
* One-Hot Encoding is doing an excellent job like Numeric Encoding, except it falls down very quickly: from 128 cardinalities, performing consistency worse than Binary Encoding. Its accuracy over all the runs is around 80%.
* Binary Encoding is very consistent in performance but not perfect, even with 8,192 cardinalities. Its exact accuracy is approximately 91%.

### Balancing Accuracy of Encodings

To look further at the details, we are splitting the balancing ratio (25% and 50%) of the positive label to check for discrepancies.

![image](https://cloud.githubusercontent.com/assets/9083669/25314687/e4e4d520-2848-11e7-968b-50a08e6c8686.png)

We are clearly noticing one major trend:
* The more the dataset is unbalanced, the more accurate the encodings become.
* Inversely: the more the dataset is balanced, the less accurate the encodings become.

We can also notice our Numeric Encoding is getting worse faster versus Binary Encoding when the dataset becomes more balanced: it becomes worse from 1024 cardinalities on a perfectly balanced dataset, while it becomes worse only from 2048 cardinalities on a 1:3 unbalanced dataset.

For One-Hot Encoding, it is even more pronounced: worse 256 cardinalities on a perfectly balanced dataset, to 128 cardinalities on a 1:3 unbalanced dataset.

> In addition, there seems to be no reason to use One-Hot Encoding over Numeric Encoding according to the picture.

In addition, there are three specific trends we can capture:

![image](https://cloud.githubusercontent.com/assets/9083669/25314690/f0d2aef2-2848-11e7-9a28-e9ba830b47b5.png)

* Numeric Encoding is not giving consistent results, as the results are flying around a bit (compare the box plot sizes and you will notice it). It seems it requires a lot more cardinalities to converge to a stable performance. It depicts also more consistency “as the balancing becomes more unbalanced”.
* One-Hot Encoding is extremely consistent. So much consistent there is not much to say about if you want to approximate the predictive power of a categorical feature when alone.
* Binary Encoding gets consistent when the cardinality increases, while maintaining a stable performance. It depicts also more consistency “as the balancing becomes more unbalanced”. Perhaps someone has a mathematical proof of convergence towards a specific limit, given the cardinality and the balancing ratio?

### Training Time

The training time is provided here as an example on dense data using rpart. Each model was run 250 times, with the median for:
* 25 times per run for Categorical Encoding.
* 25 times per run for Numeric Encoding.
* 1 time per run for One-Hot Encoding (too long…).
* 10 times per run for Binary Encoding.

It shows why you should avoid One-Hot Encoding on rpart, as the training time of the decision tree literally explodes!:

![image](https://cloud.githubusercontent.com/assets/9083669/25314696/0663dbec-2849-11e7-914d-9530f17d86e3.png)

Data is dominated by One-Hot Encoding slowness. Without One-Hot Encoding scaling issue, we have the following:

![image](https://cloud.githubusercontent.com/assets/9083669/25314700/0c18801a-2849-11e7-9c19-01da656a302c.png)

A more fair comparison.As we can clearly notice, if you are ready to spend a bit more time doing computations, then Numeric and Binary Encodings are fine.

## Conclusion (tl;dr)

A simple resume with a picture and two lines of text:

![image](https://cloud.githubusercontent.com/assets/9083669/25314703/1782a6e2-2849-11e7-8dc6-580e5f5f3aef.png)

* Categorical features with large cardinalities (over 1000): Binary
* Categorical features with small cardinalities (less than 1000): Numeric

> There seems to be no reason to use One-Hot Encoding over Numeric Encoding.
