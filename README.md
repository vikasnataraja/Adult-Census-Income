# Adult-Census-Income
This repo aims to explore various machine learning methods to predict the income for a test set by training the models on known census data.

## Formatting the dataset ##

* Before solving the actual problem itself, I used `pandas` to construct the training dataframe from the 'train.data' and 'train.demo', then used 'test.data' to construct the test dataframe.

* This function or method is under the name `formatDataset` in the code. In here, I used pandas to read the data, then converted the non-numeric data to numeric values by using certain discrete mapping values in order to use classification. It is essentially categorical encoding. I also converted the integer values to float values to make the whole dataset uniform.

* The function returns X,original dataset, and y, if applicable.

## Approach ##

Since the y variable wasn't available for the test dataset, I came up with 2 options to test my algorithms before submission - split the dataset using scikit learn's `test_train_split` or use K-Fold cross validation.

While the cross-vaidation technique seemed more obvious and better, I still tried test_train_split.

__1. Test-Train Split__

* I tested this technique on various models - RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, KNeighborsClassifier and ExtraTreeClassifier.

#### Outcome ####

* I used the same method to test all the algorithms. I used the for loop to loop through different values of the hyperparameters to tune them to the best value. Then calculated accuracy scores to compare any differences or improvements.

* However, with this method, I was only able to go slightly above 83% accuracy for the test data (which was with RandomForestClassifier). 

__2. K-Fold Cross Validation__

* This method certainly improved my accuracies. I was able to cross the 87% accuracy with more than one model. Using the documentation on cross validation from scikit-learn (https://scikit-learn.org/stable/modules/cross_validation.html), I was able to optimize the algorithms to the best accuracies.
* The split of 5 folds,`cv=5`, seemed to produce the best accuracies. 
* For every algorithm, I tuned the hyperparameters by running each feature through a for loop for various values, then optimizing each carefully, sometimes up to 9 decimal places.
* Each time a hyperparameter was optimized, the scores went up. Then repeated the same for all hyperparameters.

I used this method to test `RandomForestClassifier, DecisionTreeClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, KNeighborsClassifier and ExtraTreeClassifier`.

#### Outcome ####

* __Ultimately, the `GradientBoostingClassifier` produced the best accuracies.__
* I optimized the classifier to a high extent, testing accuracies on each change of value of a hyperparameter.

*NOTE*: I did not set the `random_state` hyperparameter intentionally. While this changed accuracies every time by a few decimal points, setting a value would've been nearly impossible since it can literally be any integer. But this also means the accuracies I obtained cannot be replicated to an exact value, but instead can be obtained to within a few decimal places.

#### Reasoning for outcome ####

* It quickly became obvious that with such variety of ranges of data, each having little or no dependence on the other features, `RandomForestClassifier` and `DecisionTreeClassifier` would be the top choices. 
* As I saturated the classifier, I realized that certain features would require ensemble methods in order to get better overall accuracies. 
* Gradient Boosting and AdaBoost both produced good accuracies with AdaBoost around 86% cross validation scores, GradientBoosting around 87%.

## What you need to run:
* Python 3
* Scikit-learn

## How to run:
* Clone the repo to your local directory
* Verify that all the files are in the same folder and there are no subfolders
* Open main.py and run

## References:
David Quigley, CU Boulder





