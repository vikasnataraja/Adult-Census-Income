#!/usr/bin/env python
# coding: utf-8

# ## Predicting Adult Census Income
# 
# Using only data from the training set, I am aiming to predict the test score (or incomes) without knowing the test
# solution. To circumvent this uncertainty, I am using cross validation which allows me to tune the hyperparameters.

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC


# ### Constructing the dataframe using pandas
# 
# * The `formatDataset` function uses pandas to create a dataframe for both test and train data.
# * I am using pandas because of the ease of use and clear labels

# In[ ]:


def formatDataset(data,listofLabels,numericLabels):
    """
    This function takes in the data(test or train), a list of labels for the features and 
    another parameter for the labels containing the numeric data (also as a list)
    """
    
    # create a dataframe using pandas, the labels are the names of features
    df = pd.DataFrame(data,columns=listofLabels)
    
    # create a duplicate, just in case
    dataset = pd.DataFrame(data,columns=listofLabels)
    
    # for the data that is already numerical, connvert to float
    dataset[numericLabels] = dataset[numericLabels].astype('float64')
    
    # get the dataframe that is not numerical
    nonNumerical = dataset.drop(columns=numericLabels)
    
    # for the non-numeric data, map the data to discrete values.
    # (I tried automatic numbering but that meant the mapping for training and testing data would be different)
    # so I created separate mapping values
    
    workclassmap = {'Private': 0,'Self-emp-not-inc': 1,'Local-gov': 2,'?': 3,
                     'State-gov': 4,'Self-emp-inc': 5,'Federal-gov': 6,'Without-pay': 7,'Never-worked': 8}
    
    educationmap = {'HS-grad': 0,'Some-college': 1,'Bachelors': 2,'Masters': 3,'Assoc-voc': 4,
                    '11th': 5,'Assoc-acdm': 6,'10th': 7,'7th-8th': 8,'Prof-school': 9,
                     '9th': 10,'12th': 11,'Doctorate': 12,'5th-6th': 13,'1st-4th': 14,'Preschool': 15}
    
    marriedmap = {'Married-civ-spouse': 0,'Never-married': 1,'Divorced': 2,'Separated': 3,'Widowed': 4,
                     'Married-spouse-absent': 5,'Married-AF-spouse': 6}
    occupationmap = {'Prof-specialty': 0,'Craft-repair': 1,'Exec-managerial': 2,'Adm-clerical': 3,
                     'Sales': 4,'Other-service': 5,'Machine-op-inspct': 6,'?': 7,'Transport-moving': 8,
                     'Handlers-cleaners': 9,'Farming-fishing': 10,'Tech-support': 11,'Protective-serv': 12,
                    'Priv-house-serv': 13,'Armed-Forces': 14}
    
    relationmap = {'Husband': 0,'Not-in-family': 1,'Own-child': 2,'Unmarried': 3,
                     'Wife': 4,'Other-relative': 5}
    
    racemap = {'White': 0,'Black': 1,'Asian-Pac-Islander': 2,'Amer-Indian-Eskimo': 3,'Other': 4}
    
    sexmap = {'Male': 0, 'Female': 1}
    
    countrymap = {'United-States': 0,'Mexico': 1,'?': 2,'Philippines': 3,'Germany': 4,'Canada': 5,
                 'Puerto-Rico': 6,'El-Salvador': 7,'India': 8,'Cuba': 9,'England': 10,'Jamaica': 11,
                 'South': 12,'China': 13,'Italy': 14,'Dominican-Republic': 15,'Vietnam': 16,'Guatemala': 17,
                 'Japan': 18,'Poland': 19,'Columbia': 20,'Taiwan': 21,'Haiti': 22,'Iran': 23,'Portugal': 24,
                  'Nicaragua': 25,'Peru': 26,'France': 27,'Greece': 28,'Ecuador': 29,'Ireland': 30,'Hong': 31,
                 'Cambodia': 32,'Trinadad&Tobago': 33,'Thailand': 34,'Laos': 35,'Yugoslavia': 36,
                  'Outlying-US(Guam-USVI-etc)': 37,'Hungary': 38,'Honduras': 39,'Scotland': 40,'Holand-Netherlands': 41}
    
    # apply the maps to the non-numeric data
    nonNumerical['workclass']=nonNumerical['workclass'].map(workclassmap)
    nonNumerical['education']=nonNumerical['education'].map(educationmap)
    nonNumerical['marital-status']=nonNumerical['marital-status'].map(marriedmap)
    nonNumerical['occupation']=nonNumerical['occupation'].map(occupationmap)
    nonNumerical['relationship']=nonNumerical['relationship'].map(relationmap)
    nonNumerical['race']=nonNumerical['race'].map(racemap)
    nonNumerical['sex']=nonNumerical['sex'].map(sexmap)
    nonNumerical['native-country']=nonNumerical['native-country'].map(countrymap)
    
    # add back the columns which were numeric in the first place
    nonNumerical.insert(loc=0,column='age',value=dataset['age'])
    nonNumerical.insert(loc=2,column='fnlwgt',value=dataset['fnlwgt'])
    nonNumerical.insert(loc=4,column='education-num',value=dataset['education-num'])
    nonNumerical.insert(loc=10,column='capital-gain',value=dataset['capital-gain'])
    nonNumerical.insert(loc=11,column='capital-loss',value=dataset['capital-loss'])
    nonNumerical.insert(loc=12,column='hours-per-week',value=dataset['hours-per-week'])
    
    # check to see if it is training data or test data
    if len(data[0])==15:
        nonNumerical.insert(loc=14,column='income',value=dataset['income'])
    
    dataset=nonNumerical
    if len(data[0])==15:
        X = dataset.drop(columns='income')
        y = dataset['income']
        
        # return X,y as well as the original dataframe for any use
        return X,y,df
    else:
        return dataset,df


# ### Pass train.data and test.data through `formatDataset` ###
# 
# * Pass the training and testing data
# * The function `formatDataset` can distinguish between them
# * It returns X,y and full dataframe in case of training set, X and full dataframe in case of testing set

# In[ ]:


"""
use the formatDataset function to construct pandas dataframe 
for both train and testing
"""

train_data = []

with open('train.data', 'r') as reading:
    train_input = reading.read().split('\n')

for row in train_input:
    train_data.append(row.split(', '))
    
train_labels = ['age','workclass','fnlwgt','education','education-num','marital-status',
                       'occupation','relationship','race','sex','capital-gain','capital-loss',
                       'hours-per-week','native-country','income'] 

train_numericLabels = ['age','fnlwgt','education-num','capital-gain',
                                    'capital-loss','hours-per-week','income']

X_train,y_train,df = formatDataset(train_data,train_labels,train_numericLabels)

# do the same for testing data, except no y for test

test_data =[]
with open('test.data', 'r') as reading_test:
    test_input = reading_test.read().split('\n')

for row in test_input:
    test_data.append(row.split(', '))


test_labels = ['age','workclass','fnlwgt','education','education-num','marital-status',
                       'occupation','relationship','race','sex','capital-gain','capital-loss',
                       'hours-per-week','native-country'] 

test_numericLabels = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']

X_test,testframe = formatDataset(test_data,test_labels,test_numericLabels)


# ### Classifiers
# 
# * Use the X and y from the previous cell to find a classifier that has the best cross-validation score.
# * I am using cross-validation because test_y is not available.
# 
# 

# In[ ]:


model = GradientBoostingClassifier(learning_rate=0.99999998,n_estimators=232,loss='exponential',
                                       max_features=8,max_depth=3,
                                       min_samples_split=9100/10000)

scores = cross_val_score(model, X_train, y_train, cv=5)
print(scores)
print(sum(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


clf = RandomForestClassifier(n_estimators=99,max_features=9,bootstrap=True,max_depth=16,
                                  random_state=42,criterion='entropy',oob_score=True)
scores = cross_val_score(clf, X_train, y_train, cv=5)
#print(i)
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


ada = AdaBoostClassifier(n_estimators=80,learning_rate=1.0,random_state=42)

scores = cross_val_score(ada, X_train, y_train, cv=5)
#print(i)
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


bag = BaggingClassifier(bootstrap=0.9,oob_score=True,n_estimators=90,max_features=12)
scores = cross_val_score(bag, X_train, y_train, cv=5)
#print(i)
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


tree = DecisionTreeClassifier(max_features=13,max_depth=9)
scores = cross_val_score(tree, X_train, y_train, cv=5)
#print(i)
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# #### Fit the model to the training set and predict y for the test set
# 
# * After finding the best model with tuning finished, fit the classifier to X and y.
# * Predict the y for the test set
# * Save the y_pred to a .csv file

# In[ ]:


model.fit(X=X_train,y=y_train)
y_pred = model.predict(X=X_test)
save_y = pd.DataFrame(y_pred,columns=['Category'])
save_y.to_csv('gradboost.csv',index=True)


# ### END OF DOCUMENT ###
