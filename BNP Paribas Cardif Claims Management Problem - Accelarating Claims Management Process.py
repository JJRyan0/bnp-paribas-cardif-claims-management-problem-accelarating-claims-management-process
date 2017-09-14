
# coding: utf-8

# ## Accelerating Insurance Claim Process with Machine Learning
# 
# ### Random Forests, Support Vector Machines, MLP Neural Networks and K- Nearest Neighbour Learning
# 
# Kaggle competition: BNP Paribas cardif claims management problem
# 
# Created by John Ryan 22th April 2017
# 
# Data source: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management

# In[1]:

from IPython.display import Image
Image("C:\\data\\image.png",  width=900, height=600)


# In[2]:

Image("C:\\data\\bnp.png",  width=900, height=600)


# In[3]:

from IPython.display import Image
Image("C:\\data\\Image2.png",  width=900, height=600)


# In[4]:

Image("C:\\data\\image1.png",  width=500, height=200)


# In[2]:

#import dependencies
import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning,
                       module="pandas, lineno=570")
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import io
import requests
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')
from sklearn import preprocessing
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


# ##   1.0 - Data Cleaning & Preperation
# 
# 1. Read in the csv file.
# 2. Describe and analysis the data
# 3. Randomization of the data, separate out all variables to make sure they have the correct     data type i.e. numeric, nominal and categorical.
# 3. Missing value treatment.
# 4. Encode labels in the data set with "one hot encoder".
# 5. Use cross Validation to create a test, training and validation data set.
# 6. Remove columns from the training data set and select important features.
# 7. Randomization of the data, separate out all variables to make sure they have the correct data type i.e. numeric, nominal and categorical.
# 8. Build machine Learning algorithms using pythons sci-kit learn from on the training, test data set constructed and preprocessed.
# 9. Evaluate the results.
# 10.Improve Performance.
# 

# In[3]:

#Load the data and make dataset a Pandas DataFrame
df = pd.read_csv("C:\\data\\claims.csv")
df.head()


# In[4]:

#summerize the data to make some initle assessments
df.describe()


# In[5]:

#A glimpse at the target variable using a bar plot
sns.countplot(x="target", data=df)


# In[6]:

#View of target varable by labels 0 = not eligible for acceleration,  
#1 = claims suitable for an accelerated approval
d = sns.countplot(x ="target", data=df)
d.set(xticklabels=['not eligible for acceleration', 'claims suitable for an accelerated approval'])
plt.xlabel(' ')
d;


# ### 1.1 - Missing Value Treatment
# 
# For this example we will use the PVI technique meaning Predictive Value Imputation. Before the model is created we will use this method when detecting missing values to estimate values that to replace NaN's with the attribute’s mean, median or mode value.
# 
# steps taken:
# 
# - Print out the number of missing values in the entire data set
# - Fill in missing value using mean value of the attributes
# - Summerize the data after cleaning or transforming activities

# In[8]:

#Print out the number of missing values
print("Number of NA values : {0}".format((df.shape[0] * df.shape[1]) - df.count().sum()))


# In[9]:

#Missing Value PVI using mean value of attributes;
x = df.fillna(df.mean())
x.head(7)


# In[11]:

print("Number of NA values : {0}".format((x.shape[0] * x.shape[1]) - x.count().sum()))


# In[12]:

#Always good preactise to summerize the data after cleaning or transforming activities
x.describe()


# ## 1.2 - Feature Importance
# ### Estimating highly predictive features with a Ensemble Random Tree Classifier
# 
# Sometimes when we are faced with a large dataset with a large amount of features it may be useful to carry out an automatic variable selection on the columns. We can make use of machine learning to make an estimate on what are the most relevant feature for a predictive model scenario. To determine a list of predictive features a meta estimator can be used that fits a number of decision trees on sub-samples of the main dataset.
# 
# The n_estimators specifies the number of randomized trees to be used. It is also important to note that the max_features parameter can also be specified to look for the best split in the data. The log2 & square root of the number of features can be used, however for this example we have used the default "auto" which uses the square root of the number of features.
# 
# Steps taken:
# 
# - Encode Labels "One Hot Encoder"
# - Assign the target variable to Y for later processing and Remove the ID Column as not required. 
# - Build ExtraTreesClassifier model to extract the best predictive features.
# - Rank the Important Variables in order
# - Subsetting the resulting data into a data frame with the top ranked features produced by the algorithim

# ### 1.2.1 - Encode Labels "One Hot Encoder"
# 
# First we need to encode any categoral attributes to numeric representations.

# In[13]:

#Label encoder tranforms any label or attribute for input to the algorithim 
#we can also see some missing values in the top few rows of the data set these will also
#need to be treated in a suitable mannor.
for feature in x.columns:
    if x[feature].dtype=='object':
        le = LabelEncoder()
        df[feature] = le.fit_transform(x[feature])
x.tail(3)


# In[14]:

#Assign the target variable to Y for later processing and 
#Remove the ID Column that is not needed 
y = x.target.values
x = x.drop(['ID'], axis = 1)


# In[15]:

#Feature Importance - selecting only highly prdictive features using random forest Model
from sklearn.ensemble import ExtraTreesClassifier
x.shape
# feature extraction
model = ExtraTreesClassifier(n_estimators = 250, max_features = "auto", random_state=0)
model.fit(x, y)
print(model.feature_importances_)


# ### 1.2.2 - Ranking Important Variables

# In[16]:

#Ranking the most imporatnt predictive variables potentially build model based on top ranked i.e 1 -16
featureimportance = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(featureimportance)[::-1]
#Print top ranked predictive featurescdata = pd.DataFrame(x, columns = [ 'target','v50','v66','v47','v110','v31','v10','v113','v114'..])
print("Feature ranking:")

for feature in range(x.shape[1]):
    print("%d. feature %d (%f)" % (feature + 1, indices[feature], featureimportance[indices[feature]]))   


# In[37]:

#Subsetting the data set using the top ranked variable produced by the algorithim
cdata = pd.DataFrame(x, columns = [ 'target','v50','v66','v47','v110','v31','v10','v114'])
cdata.tail(4)


# In[27]:

#entire dataset - removed the target variable
#X = x.drop(['target'], axis = 1)
#Y = x.target.values


# ### 2.0 - Building Machine Learning Models

# ### 2.1 - Random Forest Algorithm

# There are many advantages to applying a tree model with bootstrap aggregation and building each tree from different random subset of features which encourages ensemble diversity and thus reduces training time significantly.
# 
# Scaling or transforming the data is not necessary for random forests.
# This algorithim deals with convergence and numerical precision issues,
# which can sometimes trip up the algorithms used in logistic and linear regression, as well as neural networks, aren't so important. Because of this, you don't need to transform variables to a common scale like you might with a Neural Net.
# 
# steps taken: 
# 
# - First split the data 70% training 30% test
# - Create the Random Forest Classification Model with 60 estimators on entire dataset:
#  
#      **precision    recall  f1-score   support**
# 
#           0       0.73      0.07      0.12     10868
#           1       0.77      0.99      0.87     34861
#       avg/tot     0.76      0.77      0.69     45729
#     
# - Improve model Performance
# - Create the Random Forest Classification Model with 150 estimators, with a max depth of tree at 25 and minimum sample split 50 on subseted data with selected important features:
# 
#      **precision    recall  f1-score   support**
# 
#           0       0.65      0.16      0.25      8145
#           1       0.79      0.97      0.87     26152
# 
#        avg / tot  0.76      0.78      0.72     34297
# 
# 
# 
# 

# **Cross-Validation**

# In[38]:

#Cross - Validation - split the data into 70% training and the remainder for testing the model
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# **Building the Random Forest Model**
# 
# 

# In[45]:

#Create the Random Forest Classification Model
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=150, min_samples_split=50, max_depth=25, max_features='auto')
#Prediction on held for testing
RFpredict = RFmodel.fit(X_train, Y_train).predict(X_test)
RFpredict


# In[ ]:

#scale the data 
#from sklearn.preprocessing import StandardScaler
#scaler = preprocessing.StandardScaler().fit(x)
#X = scaler.transform(x)
#Y = cdata.target.values


# **Evaluate Model Performance - Classification Accuracy**

# In[46]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cmatrix = confusion_matrix(Y_test, RFpredict)
print (cmatrix)


# **Classification Report**
# 
# The f1-score is equal to the weighted average of the precision and recall. 
# 
# 

# In[47]:

creport1 = classification_report(Y_test, RFpredict)
print (creport1)


# -------------------------------------------------------------------------------------------
# 
# ### 2.2 K-Nearest Neighbour Classification
# 
# K-Nearest Neighbors often known as the lazy learning algorithm and one of the simple forms of classification. This method is most suitable for un-skewed data as it makes it easier to arrange data into groups. It is also suitable for establishing groups which differ but are not categorically distinct. It uses machine learning functions to group data into loosely knit categories. Each category is given a label. It does so by grouping data where there is a common pattern. It extracts sample data from the set as test data. This data is then classified. All future data is measured in relation to the test data. This is measured by way of Euclidean distance. Euclidean distance is the observed distance between two values.  A majority voting method is used to determine the closeness of the data to the test data.  The votes are weighted according to the distance between the values a neighboring value is always given a stronger weight than a value that is further away.

# Euclidean distance is specified by the following formula
# 
# $$ dist(p,q) = sqrt{(p1-q2)^2 +(p2-q2)^2+...+(pn-qn)^2} $$

# **Methodolgy**
# 
# In relation to the nearest neighbor classification, the method most used utilized in banking & Insurance to determine the granting or rejecting of loan application will be adapted and attuned to the practice of granting insurance claims. An anonymous data set was provided by BNP Paribas which includes categorical and numerical data. This will determine a likelihood of a claim being successful or not. The train dataset had 133 columns. There is a target of 1 for claims with accelerated approval to meet.  This takes into account the insurers guidelines for the awarding of claims. This data will be used to predict the probability for each claim in the test data set.  An analysis of the normalized data will be created based on nearest neighbor classifications normalization formula using MinMaxScaler(). This model will determine the suitability of nearest neighbor classification for the task described.

# **Data Preperation**
# 

# __Outlier Detection- Ensemble unsupervised learning method - Isolation Forest__
# 
# Back to ensemble trees i.e Random Forests for help!! this time we need isolation forests!
# The isolation algorithm is an unsupervised machine learning method used to detect abnormal anomalies in data such as outliers. This is once again a randomized & recursive partition of the training data in a tree structure. The number of sub samples and tree size is specified and tuned appropriately. The distance to the outlier is averaged calculating an anomaly detection score: 1 = outlier 0 = close to zero are normal data. 

# In[48]:

#A quick look a the data for outliers using a boxplot using seaborn
ax = sns.boxplot(data=cdata, orient="t", palette="Set1")


# **Cross Validation -Train/test split method:** is by far the most optimal method for training and testing a classifier to unseen data the data into 70% training and the remainder for testing the model.

# In[49]:

#Cross Validation -Train Test split method is by far the most optimal method for training and testing a classifier to unseen data the data into 70% training and the remainder for testing the model 
#using the subsetted data determined from the feature importance stage
X = cdata
Y = cdata.target.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# ### 2.2.1 - Scaling Data - Min Max Scaler
# Subtracts the minimum of feature X from each value and divide by the range of X.or min - max scaling in sklearn data needs to be a numpy array.

# In[51]:

#subtracts the minimum of feature X from each value and divide by the range of X.
from sklearn.preprocessing import MinMaxScaler
X= np.array(X)#for min - max scaling in sklearn data needs to be a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
rescaled_cdata = min_max_scaler.fit_transform(X)


# In[52]:

rescaled_cdata


# #### 2.2.2 - Cross - Validation on rescaled data for KNN
# 
# For this model we have we will used our sub sample date created during our feature importance stage of the analysis, the dataframe produced is "cdata". There is a fine balance between over-fitting and under-fitting the training data with this model which is sometimes reffered to as the "bias-variance trade-off". Assigning a large K will reduce this effect caused by noise.

# In[53]:

#Cross - Validation 
from sklearn import cross_validation
x = rescaled_cdata
y = cdata.target.values
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)


# ### 2.2.3 - Estimating the number of Neighbours
# 
# In practice, choosing k depends on the difficulty of the concept to be learned and the number of records in the training data.
# Typically, k is set somewhere between 3 and 10.
# One common practice is to set k equal to the square root of the number of training examples.

# In[54]:

#get length of training data to determine the number of neighbours to select for KNN Model
len(x_train)


# In[55]:

#standard use square root of training data length for n_neighbours = 240
import cmath
trainlen = 57160
train_sqrt = cmath.sqrt(trainlen)
print (train_sqrt)


# $$n = \sqrt{57160} = 239.08157$$

# ### 2.2.4 - Build the KNN Model

# In[64]:

#Model 1 - Training K Nearest Neighbour Classification
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors= 239,weights='uniform')
y_pred2 = clf.fit(x_train, y_train).predict(x_test)
y_pred2


# **Evaluating Model Performance - Classification Accuracy**

# **Classification Report**

# In[68]:

#Evaluating KNN model performance
creport2 = classification_report(y_test, y_pred2)
print (creport2)


# Data source: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
