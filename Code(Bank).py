# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:34:53 2019

@author: Amol
"""

# Aim: The aim is to build a data model to predict the probability of default in the future.

# Group memners:
# 1. Shubham Chaudhari
# 2. Amol Shinde
#%%
# importing the libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from scipy.stats import boxcox
#%%

# reading the files
df = pd.read_csv("E:\Imarticus\Final Project\Python\Dataset\XYZCorp_LendingData.txt",delimiter='\t', header =0) 

df.head()
#%%
# check for shape
print(df.shape)
#%%

# setting up the column width to ensure that all columns are displayed
pd.set_option('display.max_columns',None)

#%%
# information of columns
df.info()

#%%
# knowing the statistical values of the data
df.describe()

# datatypes of the features
df.dtypes

#%%
# copy of data in new dataframe for operating on data
loan_df = df.copy ()

#%%
# Finding the the count and percentage of values that are missing in the dataframe.
loan_df_null = pd.DataFrame({'Count': loan_df.isnull().sum(), 'Percent': 100*loan_df.isnull().sum()/len(loan_df)})

#%%
# printing columns with null count more than 0
loan_df_null[loan_df_null['Count'] > 0] 

#%%
# remove null values or columns having missing values greater than 80%
loan_df1 = loan_df.dropna(axis=1, thresh=int(0.80*len(loan_df)))

#%%
# exploratory data analysis

# we will now select the columns that are necessary for our analysis

loan_df2 = loan_df1.filter(['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership',
                    'annual_inc','verification_status','purpose','dti','delinq_2yrs','loan_status'])
loan_df2.dtypes

#%%
# finding the correlation between variables
plt.figure(figsize=(20,20))
sns.set_context("paper", font_scale=1)

#finding the correllation matrix and changing the categorical data to category for the plot.
sns.heatmap(loan_df2.assign(grade=loan_df2.grade.astype('category').cat.codes,
                         sub_g=loan_df2.sub_grade.astype('category').cat.codes,
                         term=loan_df2.term.astype('category').cat.codes,
                        emp_l=loan_df2.emp_length.astype('category').cat.codes,
                         ver =loan_df2.verification_status.astype('category').cat.codes,
                        home=loan_df2.home_ownership.astype('category').cat.codes,
                        purp=loan_df2.purpose.astype('category').cat.codes).corr(), 
                         annot=True, cmap='bwr',vmin=-1, vmax=1, square=True, linewidths=0.5)

loan_df2.drop(['installment','grade','sub_grade','verification_status','term'], axis=1, inplace = True)

#%%
# print the count and null values in the dataframe
loan_df_null1 = pd.DataFrame({'Count': loan_df2.isnull().sum(), 'Percent': 100*loan_df2.isnull().sum()/len(loan_df2)})
loan_df_null1[loan_df_null1['Count'] > 0]

#%%
# dropping the null rows since we have sufficient amount of data and there is no need to fill the null values.
loan_df2.dropna(axis=0)

#%%
# printing unique statuses in the loan status column (dependent variable)
loan_df2['default_ind'].unique()

#%%
# distribution of the default_ind values
a =loan_df2['default_ind'].value_counts()
a = a.to_frame()
a.reset_index(inplace=True)
a.columns = ['default_ind','Count']
plt.subplots(figsize=(20,8))
sns.barplot(y='Count', x='default_ind', data=a)
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Loan Status in our Dataset")
plt.show()

#%%
# loan_amnt column
plt.figure(figsize=(12,6))
plt.subplot(121)
g = sns.distplot(loan_df2["loan_amnt"])
g.set_xlabel("Loan Amount", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition", fontsize=20)
plt.show()

#%%
# Home Ownership v/s Loan Amount
plt.figure(figsize = (9,6))
g = sns.violinplot(x="home_ownership",y="loan_amnt",data=loan_df2,
               kind="violin",
               split=True,palette="hls",
               hue="application_type")
g.set_title("Homer Ownership - Loan Distribuition", fontsize=20)
g.set_xlabel("Homer Ownership", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

#%%
# Disperssion Plot of Insatllment
plt.figure(figsize=(10,6))
sns.distplot(loan_df2['installment'])
plt.show()

#%%
# Data Visulization of Int_rate
loan_df2['int_round'] = loan_df2['int_rate'].round(0).astype(int)
plt.figure(figsize = (10,8))

plt.subplot(211)
g = sns.distplot(np.log(loan_df2["int_rate"]))
g.set_xlabel("", fontsize=12)
g.set_ylabel("Distribuition", fontsize=12)
g.set_title("Int Rate Log distribuition", fontsize=20)

plt.subplot(212)
plt.figure(figsize=(10,6))
g1 = sns.countplot(x="int_round",data=loan_df2, 
                   palette="Set2")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Int Rate Normal Distribuition", fontsize=20)
plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()

#%%
# Loan Applied by borrower vs Amt Funded by Lender vs Total committed by investor 
fig, ax = plt.subplots(1, 3, figsize=(16,5))

loan_amnt = loan_df["loan_amnt"].values
funded_amnt = loan_df["funded_amnt"].values
funded_amnt_inv = loan_df["funded_amnt_inv"].values

sns.distplot(loan_amnt, ax=ax[0], color="#F7522F")
ax[0].set_title("Loan Applied by the Borrower", fontsize=14)
sns.distplot(funded_amnt, ax=ax[1], color="#2F8FF7")
ax[1].set_title("Amount Funded by the Lender", fontsize=14)
sns.distplot(funded_amnt_inv, ax=ax[2], color="#2EAD46")
ax[2].set_title("Total committed by Investors", fontsize=14)

#%%
loan_df2['default_ind'].value_counts()
loan_df2.dtypes
#%%
# transformation
loan_df2.issue_d=pd.to_datetime(loan_df2.issue_d)
loan_df2.shape

numerical = loan_df2.columns[loan_df2.dtypes == 'float64']
for i in numerical:
    if loan_df2[i].min() > 0:
        transformed, lamb = boxcox(loan_df2.loc[df[i].notnull(), i])
        if np.abs(1 - lamb) > 0.02:
            loan_df2.loc[df[i].notnull(), i] = transformed

#%%
# one hot encoding 
loan_df2 = pd.get_dummies(loan_df2, drop_first=True)
loan_df2.shape

#%%
# split into train and test
traindata= loan_df2[loan_df2['issue_d'] <= '2015-05-01']
traindata.shape
traindata.sample
print(traindata.head(n=3))

testdata= loan_df2[loan_df2['issue_d'] > '2015-05-01']
testdata.shape
print(testdata.head(n=30))

#%%
# let's look at the summary statistics of the issue dates in the train and test sets:
traindata['issue_d'].describe()
testdata['issue_d'].describe()

#%%
# drop issue_d
traindata.drop('issue_d', axis=1, inplace=True)
testdata.drop('issue_d', axis=1, inplace=True)

testdata.reset_index(drop=True, inplace=True)
traindata.reset_index(drop=True, inplace=True)

#%%
# scale the data so that each column has a mean of zero and unit standard deviation
sc = StandardScaler()
X_train = traindata.drop('default_ind', axis=1)
Y_train = traindata['default_ind']
numerical = X_train.columns[(X_train.dtypes == 'float64') | (X_train.dtypes == 'int64')].tolist()
X_train[numerical] = sc.fit_transform(X_train[numerical])

#%%
# model selection
def createROC(models, X, y, X_test, Y_test):
    false_p, true_p = [], [] #false postives and true positives

    for i in models.keys():  #dict of models
        models[i].fit(X, y)

        fp, tp, threshold = roc_curve(Y_test, models[i].predict_proba(X_test)[:,1]) ##roc_curve function

        true_p.append(tp)
        false_p.append(fp)
    return true_p, false_p #returning the true postive and false positive

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

models = {'MNB': MultinomialNB(),
          'RF': RandomForestClassifier(n_estimators=100),
          'LR': LogisticRegression(C=1)}

setA = {}
for i in models.keys():
    scores = cross_val_score(models[i], X_train - np.min(X_train) + 1,
                                    Y_train, cv=3)
    setA[i] = scores
    print(i, scores, np.mean(scores))

#%%
# test set analysis and scaling
X_test = testdata.drop('default_ind', axis=1)
Y_test = testdata['default_ind']
numerical = X_test.columns[(X_test.dtypes == 'float64') | (X_test.dtypes == 'int64')].tolist()
X_test[numerical] = sc.fit_transform(X_test[numerical])

#%%
# computing ROC curves for models
tp_unbalset, fp_unbalset = createROC(models, X_train - np.min(X_train) + 1, Y_train, X_test - np.min(X_test) + 1, Y_test)

#%%
# fitting LR to the test data set
model =  LogisticRegression(C=1)
model.fit(X_train, Y_train)
predict = model.predict(X_test) #prediction of X_test which can be used to test against Y_test (testdata values or true values of y)

a = Y_test.to_frame()
a['default_ind'].value_counts()

#%%
# cv-score, ROC curve, CM, of random forest
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,5))

ax = pd.DataFrame(setA).boxplot(widths=(0.9,0.9,0.9,0.9), grid=False, vert=False, ax=axes[0])
ax.set_ylabel('Classifier')
ax.set_xlabel('Cross-Validation Score')

for i in range(0, len(tp_unbalset)):
    axes[1].plot(fp_unbalset[i], tp_unbalset[i], lw=1)

axes[1].plot([0, 1], [0, 1], '--k', lw=1)
axes[1].legend(models.keys())
axes[1].set_ylabel('True Positive Rate')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_xlim(0,1)
axes[1].set_ylim(0,1)

cm = confusion_matrix(Y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

ax = sns.heatmap(cm, annot=True, cmap='Blues', ax=axes[2]);
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.axis('equal')

#%%
# getting false and true positive from test set
fp, tp, threshold = roc_curve(Y_test, model.predict_proba(X_test)[:,1]) #getting false and true positive from test set

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

ax[0].plot(threshold, tp + (1 - fp))
ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('Sensitivity + Specificity')

ax[1].plot(threshold, tp, label="tp")
ax[1].plot(threshold, 1 - fp, label="1 - fp")
ax[1].legend()
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('True Positive & False Positive Rates')

#%%
# finding the optimal threshold for the model 
function = tp + (1 - fp)
index = np.argmax(function)

optimal_threshold = threshold[np.argmax(function)]
print('optimal threshold:', optimal_threshold)

#%%
# using threshold for the model
predict = model.predict_proba(X_test)[:,1]
predict = np.where(predict >= optimal_threshold, 1, 0)

fig, axes = plt.subplots(figsize=(15,6))

cm = confusion_matrix(Y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

ax = sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.axis('equal')

#%%
# balancing the training dataset and creating a new model
Y_default = traindata[traindata['default_ind'] == 0]
n_paid = traindata[traindata['default_ind'] == 1].sample(n=len(Y_default), random_state=17) # chosing equal amount of 1's

#%%
# creating a new dataframe for balanced set
data = Y_default.append(n_paid) 

#%%
# creating the independent and dependent array
X_bal = data.drop('default_ind', axis=1)
y_bal = data['default_ind']

#%%
# scaling it again
numerical = X_bal.columns[(X_bal.dtypes == 'float64') | (X_bal.dtypes == 'int64')].tolist()
X_bal[numerical] = sc.fit_transform(X_bal[numerical])

#%%
# training the model on the balanced set
models = {'MNB': MultinomialNB(),
          'RF': RandomForestClassifier(n_estimators=100),
          'LR': LogisticRegression(C=1)}

balset = {}
for i in models.keys():
    scores = cross_val_score(models[i], X_bal - np.min(X_bal) + 1,
                                    y_bal, scoring='roc_auc', cv=3)
    balset[i] = scores
    print(i, scores, np.mean(scores))

#%%
# predictions on random forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_bal, y_bal)
predict = model.predict(X_test)

predict = model.predict(X_test)
fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(Y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)
ax = sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.axis('equal')

#%%
# finding optimum number of estimators
params = {'n_estimators': [50, 100, 200, 400, 600, 800]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params,
                                   scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_bal, y_bal)
print(grid_search.best_params_)
print(grid_search.best_score_)

#%%
# using estimator to fit the model
grid_search.best_estimator_.fit(X_bal, y_bal)
predict = model.predict(X_test)

fig, axes = plt.subplots(figsize=(15,9))
cm = confusion_matrix(Y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)
ax = sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.axis('equal')

#%%
# Plotting important variables
r = pd.DataFrame(columns=['Feature','Importance'])
ncomp = 15
r['Feature'] = feat_labels = X_bal.columns
r['Importance'] = model.feature_importances_
r.set_index(r['Feature'], inplace=True)
ax = r.sort_values('Importance', ascending=False)[:ncomp].plot.bar(width=0.9, legend=False, figsize=(15,8))
ax.set_ylabel('Relative Importance')
#%%