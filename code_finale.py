#!/usr/bin/env python
# coding: utf-8

#  # Data preprocessing

# ### Datas importation and visualisation

# In[267]:


#project machine learning tyres train 
#load the data

import pandas as pd
import numpy as np
data = pd.read_csv("tyres_train.csv")


# In[268]:


data.sample(10)


# In[269]:


#check data size
print(data.shape)


# In[270]:


print(type(data))


# In[271]:


#get datas columns
data.columns


# ### Checking duplicated datas

# In[272]:


#search for dublicates
data.duplicated()


# In[273]:


data[data.duplicated()]


# In[274]:


print(data[data.duplicated()].shape)
#no data duplicated


# ### Processsing incompletness of the datas

# In[275]:


#checking missing values
data.isnull()


# In[276]:


data.isna().any()


# In[277]:


#There are missing in the diameter column
data[data["diameter"].isna()]


# In[278]:


#Amount of missing data
print(data[data["diameter"].isna()].shape)


# In[279]:


#We choose to drop diameter column
data = data.dropna(axis=1) # or axis=1 #remove the  column
data


# ### Data exploration

# In[280]:


##matplotlib inline
##histogram of the attributes for both target values

import seaborn as sns
import matplotlib.pyplot as plt


X0 = data[data['failure']==0]
X1 = data[data['failure']==1]

# fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(20,15))
# fig.tight_layout()

# for i, ax in zip(range(data.columns.size), axes.flat):
#     sns.histplot(X0.iloc[:,i], color="blue", element="step", ax=ax,  alpha=0.3)
#     sns.histplot(X1.iloc[:,i], color="red", element="step", ax=ax,  alpha=0.3)
# plt.show()


# In[281]:


##boxplot of the attributes for both target values

# fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(12,15))
# fig.tight_layout(pad=2)

# for i, col in enumerate(data.columns[:-1]):
#     sns.boxplot(y = col, x = "failure",data=data, orient='v', ax=axes[int(i/5),i%5])


# In[282]:


# #sns plot of attributes

# sns.pairplot(data, hue='failure')


# ### Data normalisation distribution

# In[283]:


# #checking whether the histogramm is closed to a normal distribution


# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

X=data

# fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(20,15))
# fig.tight_layout()

# for i, ax in zip(range(X.columns.size), axes.flat):
#     sns.histplot(X.iloc[:,i], color="blue", element="step", ax=ax,  alpha=0.3)
# plt.show()


# In[284]:


import math
pd.options.mode.chained_assignment = None # no warning on creating a new column

#creating a dataframe of the logarithm of the attributes
#removing temperature because it is negative 

min_temp = min(data['temperature'])

numerical=['vulc', 'perc_nat_rubber', 'weather', 'perc_imp', 'elevation', 'perc_exp_comp']
categorical = ['tread_type','tread_depth','month','wiring_strength','add_layers','tyre_season','tyre_quality','failure']

data_temp_pos = data.copy()
data_temp_pos = data_temp_pos[numerical]
data_temp_pos['temperature'] = data['temperature']- min_temp+1

data_log = data_temp_pos.copy()

numerical.append('temperature')

for i in numerical:
    data_log[('log%s' %(i))]=data_temp_pos[('%s'%(i))].apply(lambda x: math.log(x+1))
    
data_log.drop(columns= numerical,inplace = True)
data_log[categorical] = data[categorical]

data_log


# data_log = data_no_temp_log.copy()
# data_log['temperature'] = data['temperature']
# data_log
# data_no_temp_log.drop(columns= column,inplace=True)
# data_log['temperature'] = data['temperature']
# data_log.drop(columns= column,inplace=True)


# In[285]:


# Plotting the log and non log data to inspect the difference
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))

plt.subplot(622)
plt.hist(data_log['logvulc'])
plt.title('logvulc')

plt.subplot(621)
plt.hist(data['vulc'])
plt.title('vulc')

plt.subplot(624)
plt.hist(data_log['logtemperature'])
plt.title('logtemperature')

plt.subplot(623)
plt.hist(data['temperature'])
plt.title('temperature')


plt.subplot(626)
plt.hist(data_log['logelevation'])
plt.title('logelevation')

plt.subplot(625)
plt.hist(data['elevation'])
plt.title('elevation')


plt.subplot(628)
plt.hist(data_log['logperc_nat_rubber'])
plt.title('logperc_nat_rubber')

plt.subplot(627)
plt.hist(data['perc_nat_rubber'])
plt.title('perc_nat_rubber')


plt.subplot(6,2,10)
plt.hist(data_log['logperc_exp_comp'])
plt.title('logperc_exp_comp')

plt.subplot(6,2,9)
plt.hist(data['perc_exp_comp'])
plt.title('perc_exp_comp')


plt.subplot(6,2,12)
plt.hist(data_log['logweather'])
plt.title('logweather')

plt.subplot(6,2,11)
plt.hist(data['weather'])
plt.title('weather')


# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()


# ### Data target distribution

# In[286]:


print('non target 0: ', X0.shape[0])

print('target 1: ',X1.shape[0])

sns.countplot(y=X.failure ,data=X) #"target" is the name of the target column, change it accordingly to your dataset
plt.xlabel("count of each class")
plt.ylabel("classes")
plt.show()


# In[287]:


#data are imbalanced, work in section 'data imbalance'


# ### Data outliers processing 

# In[288]:


data_wout_out = data.copy()
 
''' Detection of outliers'''
# this method is adapted for every type of data (even non gaussian data which is the case here) 
# liste 
#Liste = ['vulc', 'elevation','perc_exp_comp','temperature']
Liste = ['vulc','perc_nat_rubber','weather','perc_imp','temperature','elevation','perc_exp_comp']
# IQR

for name in Liste:
    print(name)
    Q1 = np.percentile(data_wout_out[name], 25,interpolation = 'midpoint')

    Q3 = np.percentile(data_wout_out[name], 75,interpolation = 'midpoint')
    IQR = Q3 - Q1
 
    print("Old Shape: ", data_wout_out.shape)
 
    # Upper bound
    upper = np.where(data_wout_out[name] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(data_wout_out[name] <= (Q1-1.5*IQR))

    ''' Removing the Outliers '''
    data_wout_out.drop(upper[0], inplace = True)
    data_wout_out.drop(lower[0], inplace = True)
    
    data_wout_out.index = range(len(data_wout_out))

    print("New Shape: ", data_wout_out.shape)


# In[289]:


#detecting outliers and removing them
#several techniques are implemented 

#removing them

import sklearn
import pandas as pd
 

data_wout_out_log = data_log.copy()
 
''' Detection of outliers'''
# this method is adapted for every type of data (even non gaussian data which is the case here) 

Liste = ['logvulc', 'logelevation','logperc_exp_comp','logperc_nat_rubber','logweather','logtemperature','logperc_imp']
# IQR

for name in Liste:
    print(name)
    Q1 = np.percentile(data_wout_out_log[name], 25,interpolation = 'midpoint')

    Q3 = np.percentile(data_wout_out_log[name], 75,interpolation = 'midpoint')
    IQR = Q3 - Q1
 
    print("Old Shape: ", data_wout_out_log.shape)
 
    # Upper bound
    upper = np.where(data_wout_out_log[name] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(data_wout_out_log[name] <= (Q1-1.5*IQR))

    ''' Removing the Outliers '''
    data_wout_out_log.drop(upper[0], inplace = True)
    data_wout_out_log.drop(lower[0], inplace = True)
    
    data_wout_out_log.index = range(len(data_wout_out_log))

    print("New Shape: ", data_wout_out_log.shape)


# In[290]:


#the log transformation seems to reduce a little bit the log transformation but not a lot 
data_wout_out_log


# In[291]:


# third method data winsowrize 

import sklearn
import pandas as pd
from scipy.stats.mstats import winsorize

column = data.columns
n = len(column)
m = len(data['vulc'])
data_winsorized = np.zeros((m,n))
for i in range(n):
     data_winsorized[:,i] =  winsorize(data[column[i]],(0.01,0.02))

        
data_winsor = pd.DataFrame(data = data_winsorized , index = np.arange(0,m), columns = column )
data_winsor.head()
#sns.boxplot(x = data_winsorized)


# ### Data standardization

# In[292]:


#data standardization with standardscaler, and other scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
numerical_attributes = ['vulc','perc_nat_rubber','weather','perc_imp','temperature','elevation','perc_exp_comp']
numerical_attributes_log = ['logvulc','logperc_nat_rubber','logweather','logperc_imp','logtemperature','logelevation','logperc_exp_comp']
Numericals_1 = data[numerical_attributes]
Numericals_2 = data_wout_out[numerical_attributes]
Numericals_3 = data_wout_out_log[numerical_attributes_log]
Numericals_4 = data_winsor[numerical_attributes]
Numericals_5 = data_log[numerical_attributes_log]
# scaler1 = StandardScaler()
# scaler2 = StandardScaler()
# scaler3 = StandardScaler()
# scaler4 = StandardScaler()
# scaler5 = StandardScaler()
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
scaler4 = MinMaxScaler()
scaler5 = MinMaxScaler()
scaler1.fit(Numericals_1) 
scaler2.fit(Numericals_2)
scaler3.fit(Numericals_3)
scaler4.fit(Numericals_4)
scaler5.fit(Numericals_5)


# In[293]:


# Save the scaler
import pickle
pickle.dump(scaler1, open('scaler.pkl', 'wb'))


# In[294]:


scaled_data_1 = scaler1.transform(Numericals_1)
scaled_data_2 = scaler2.transform(Numericals_2)
scaled_data_3 = scaler3.transform(Numericals_3)
scaled_data_4 = scaler4.transform(Numericals_4)
scaled_data_5 = scaler5.transform(Numericals_5)
#The scaler instance can then be used on new data (e.g.TEST SET!)


# In[295]:


scaled_df_1 = pd.DataFrame(scaled_data_1)
scaled_df_1.columns = numerical_attributes

scaled_df_2 = pd.DataFrame(scaled_data_2)
scaled_df_2.columns = numerical_attributes

scaled_df_3 = pd.DataFrame(scaled_data_3)
scaled_df_3.columns = numerical_attributes_log

scaled_df_4 = pd.DataFrame(scaled_data_4)
scaled_df_4.columns = numerical_attributes

scaled_df_5 = pd.DataFrame(scaled_data_5)
scaled_df_5.columns = numerical_attributes_log


# In[296]:


#Covariance function, to look if any attributes are deeply correlated
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (5,5))
sns.heatmap(data=scaled_df_1.corr().round(2), cmap='coolwarm', linewidths=.5, annot=True, annot_kws={"size":12})
plt.show()


# In[297]:


# visualising the distributions of the data after different preprocessing 
scaled_df_1.boxplot()


# In[298]:


scaled_df_2.boxplot()


# In[299]:


scaled_df_3.boxplot()


# In[300]:


scaled_df_4.boxplot()


# ### Data categorical converting

# In[301]:


#conversion of categorical attributes that are non binary (nb) using get_dummies
data_categorical_nb_1=data[['tread_type','tread_depth','month','wiring_strength','add_layers']]
data_categorical_nb_2=data_wout_out[['tread_type','tread_depth','month','wiring_strength','add_layers']]
data_categorical_nb_3=data_wout_out_log[['tread_type','tread_depth','month','wiring_strength','add_layers']]
data_categorical_nb_4=data_winsor[['tread_type','tread_depth','month','wiring_strength','add_layers']]
data_categorical_nb_5=data_log[['tread_type','tread_depth','month','wiring_strength','add_layers']]
#data_categorical_nb_1.head()
#data_categorical_nb_2.head()


# In[302]:


#creating new column with get_dummies
dummies_1 = pd.get_dummies(data_categorical_nb_1, columns=['tread_type','tread_depth','month','wiring_strength','add_layers']) 

dummies_1.tail()

print(dummies_1.shape)


dummies_2 = pd.get_dummies(data_categorical_nb_2, columns=['tread_type','tread_depth','month','wiring_strength','add_layers']) 

dummies_2.tail()

print(dummies_2.shape)


dummies_3 = pd.get_dummies(data_categorical_nb_3, columns=['tread_type','tread_depth','month','wiring_strength','add_layers']) 


dummies_3.tail()

print(dummies_3.shape)


dummies_4 = pd.get_dummies(data_categorical_nb_4, columns=['tread_type','tread_depth','month','wiring_strength','add_layers']) 

dummies_4.tail()

print(dummies_4.shape)


dummies_5 = pd.get_dummies(data_categorical_nb_5, columns=['tread_type','tread_depth','month','wiring_strength','add_layers']) 

dummies_5.tail()

print(dummies_5.shape)


# ###  Creating the differents possible datasets

# In[303]:


#Cells where we register all the possible set we will work on


#df_1

#data_transformation : no 
#outliers_removing : no
#winsorizing : no
#standardization : standardscalar 


#concatenate dummies with the other attributes
Col = ['tyre_season','tyre_quality','failure']


# X are the final data that we are going to use 

X=pd.concat([dummies_1,scaled_df_1,data[Col]], axis = 1)
X.tail()

print(X.shape)

#X is the good dataset for training part but not for data exploration

df_1 = X


# In[304]:


#df_2

#data_transformation : no 
#outliers_removing : yes
#winsorizing : no
#standardization : standardscalar 

#concatenate dummies with the other attributes
Col = ['tyre_season','tyre_quality','failure']


# X are the final data that we are going to use 

X=pd.concat([dummies_2,scaled_df_2,data_wout_out[Col]], axis = 1)
X.tail()

print(X.shape)

#X is the good dataset for training part but not for data exploration


df_2 = X
df_2


# In[305]:


#df_3

#data_transformation : logarithm
#outliers_removing : yes
#winsorizing : no
#standardization : standardscalar 

#concatenate dummies with the other attributes
Col = ['tyre_season','tyre_quality','failure']


# X are the final data that we are going to use 

X=pd.concat([dummies_3,scaled_df_3,data_wout_out_log[Col]], axis = 1)
X.tail()
print(X.shape)

#X is the good dataset for training part but not for data exploration


df_3 = X

df_3


# In[306]:


#df_4

#data_transformation : no
#outliers_removing : no
#winsorizing : yes
#standardization : standardscalar 

#concatenate dummies with the other attributes
Col = ['tyre_season','tyre_quality','failure']


# X are the final data that we are going to use 

X=pd.concat([dummies_4,scaled_df_4,data_winsor[Col]], axis = 1)
X.tail()

print(X.shape)

#X is the good dataset for training part but not for data exploration

df_4 = X


# In[307]:


#df_5

#data_transformation : logarithm
#outliers_removing : no
#winsorizing : no
#standardization : standardscalar 

#concatenate dummies with the other attributes
Col = ['tyre_season','tyre_quality','failure']


# X are the final data that we are going to use 

X=pd.concat([dummies_5,scaled_df_5,data_log[Col]], axis = 1)
X.tail()
print(X.shape)




df_5 = X

df_5


# ### Spliting training and test set 

# In[ ]:





# In[308]:


#splitting training and testing set 


#Separate X and y (explanatory variables and target variable)

# Don't forget to choose you df


X = df_1.iloc[:,:-1]
y = df_1.iloc[:,-1] #[-1]]

#X.head()
y.head()



# In[309]:


from sklearn.model_selection import train_test_split

#SPLIT DATA INTO TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size =0.30, #by default is 75%-25%
                                                    #shuffle is set True by default,
                                                    stratify=y,
                                                    random_state= 123
                                                   ) #fix random seed for replicability

print(X_train.shape,X_test.shape)

y_train


# ### Data imbalanced 

# ### Oversampling the train set (optional)

# In[311]:


#from imblearn.over_sampling import RandomOverSampler

# example of random oversampling to balance the class distribution
#from collections import Counter
# define oversampling strategy
#oversample = RandomOverSampler(sampling_strategy='minority')


# fit and apply the transform
#X_over, y_over = oversample.fit_resample(X_train, y_train)
#print(Counter(y_over))

#uncomment if you want to use it 

# X_train = X_over
# y_train = y_over


# ### Undersampling the train set (optional)

# In[312]:


from imblearn.under_sampling import RandomUnderSampler 

# example of random oversampling to balance the class distribution
from collections import Counter
# define oversampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')


...
# fit and apply the transform
X_under, y_under = undersample.fit_resample(X_train, y_train)
print(Counter(y_under))

#uncomment if you want to use it 

# X_train = X_under
# y_train = y_under


# ### PCA (optional) 

# In[313]:


#PCA fit
#we use a PCA transformation on our data set, and then we set the varaince treshold to 0,90, so when the princibel components can be explain 0,9 % of the data, the model will say:

from sklearn.decomposition import PCA
pca = PCA(n_components = 0.90)
pca.fit(X_train)
print("Cumulative Variances (Percentage):")
print(np.cumsum(pca.explained_variance_ratio_ * 100))
components = len(pca.explained_variance_ratio_)
print(f'Number of components: {components}')
#Make the scree plot
plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")


# In[314]:


#remove rest of the rest of principal components in both test and validation

#uncomment if you want to use it

# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)


# # Classification

# In[315]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


# In[316]:


# KNN
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
parameters = {'n_neighbors':np.arange(10,500,20)}

#Defining the gridsearch, and using this hyperp_search for every model, finding the f1 score. 
def hyperp_search(classifier, parameters):
    gs = GridSearchCV(classifier, parameters, cv=3, scoring = 'f1', verbose=0, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print("f1_train: %f using %s" % (gs.best_score_, gs.best_params_))

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    print("f1         train %.3f   test %.3f" % (f1_score(y_train, y_pred_train), f1_score(y_test, y_pred) )) 
    print("precision  train %.3f   test %.3f" % (precision_score(y_train, y_pred_train), precision_score(y_test, y_pred) )) 
    print("")
    print(confusion_matrix(y_test, y_pred))
    return  f1_score(y_test, y_pred)
    #print(classification_report(y_test, y_pred))


# In[317]:


f1_KNN = hyperp_search(classifier,parameters)


# In[318]:


model_knn = KNeighborsClassifier(n_neighbors=30)

def roc(model,X_train,y_train,X_test,y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_probs = model.predict_proba(X_test) #predict_proba gives the probabilities for the target (0 and 1 in your case) 

    fpr, tpr, thresholds1=metrics.roc_curve(y_test,  y_probs[:,1])

    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    auc = metrics.roc_auc_score(y_test, y_probs[:,1])
    print('AUC: %.2f' % auc)
    return (fpr, tpr)

fpr1,tpr1=roc(model_knn,X_train,y_train,X_test,y_test)


# In[319]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


classifier = RandomForestClassifier(n_estimators = 100)
parameters = {'criterion': ['entropy','gini'], 
              'max_depth': [4,5,6,8,10],
              'min_samples_split': [5,10,20],
              'min_samples_leaf': [5,10,20],
             "class_weight": ['balanced',None]}


f1_random_for = hyperp_search(classifier,parameters)


#fpr2,tpr2=roc(model_tree,X_train,y_train,X_test,y_test)


# In[320]:


model_random_for = RandomForestClassifier(n_estimators = 100)
fpr,tpr = roc(model_random_for,X_train,y_train,X_test,y_test)


# In[321]:


#Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
parameters = {'criterion': ['entropy','gini'], 
              'max_depth': [4,5,6,8,10],
              'min_samples_split': [5,10,20],
              'min_samples_leaf': [5,10,20],
             "class_weight": ['balanced',None]}

f1_tree = hyperp_search(classifier,parameters)



# In[322]:


model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, min_samples_split=20)

fpr2,tpr2=roc(model_tree,X_train,y_train,X_test,y_test)


# In[323]:


from sklearn import tree
r = tree.export_text(model_tree,feature_names=X_test.columns.tolist(),max_depth=4)
print(r)


# In[324]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB #or alternative NB implementations

model = GaussianNB()

model.fit(X_train, y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test, y_pred))

print("f1_test: ", f1_score(y_test, y_pred))

f1_bayes = f1_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[325]:


y_probs = model.predict_proba(X_test) #predict_proba gives the probabilities for the target (0 and 1 in your case) 

fpr3,tpr3=roc(model,X_train,y_train,X_test,y_test)


# In[326]:


# Logistic

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
parameters = {"C":[1e-4,1e-3,1e-2,1e-1,1,10], "max_iter":[1000],"class_weight": ['balanced',None] }


f1_log_reg = hyperp_search(classifier,parameters)


# In[327]:


model = LogisticRegression(C=10, max_iter=1000)

fpr4,tpr4=roc(model,X_train,y_train,X_test,y_test)


# In[328]:


model.fit(X_train,y_train)

coeff=pd.DataFrame()
coeff["feature"]=X_train.columns
coeff["w"]=model.coef_[0]

coeff.sort_values(by=['w'], inplace=True)


# In[329]:


#showing weights and inpact on target possibility
sns.set(rc={'figure.figsize':(10,10)})
sns.barplot(data=coeff, y="feature", x="w", palette="Blues_d", orient="h")
sns.set(rc={'figure.figsize':(6,4)})


# In[330]:


#SVM

from sklearn.svm import SVC

classifier = SVC()
parameters = {"kernel":['linear','rbf'], "C":[0.01,0.1,100],"class_weight": ['balanced',None]}

f1_SVC = hyperp_search(classifier,parameters)


# In[331]:


model_SVC = SVC(C=100, kernel='linear',class_weight='balanced',probability=True)

fpr5,tpr5=roc(model_SVC,X_train,y_train,X_test,y_test)


# In[332]:


# Multi-layer Perceptron classifier

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
parameters = {"hidden_layer_sizes":[(10, 5),(100,20,5)],  "max_iter": [2000], "alpha": [0.001,0.1]}

f1_neu_net = hyperp_search(classifier,parameters)


# In[333]:


model_MLP=MLPClassifier(hidden_layer_sizes=(10,5), alpha=0.1, max_iter=2000)

fpr6,tpr6=roc(model_MLP,X_train,y_train,X_test,y_test)


# In[334]:


# Comparring all the models:

plt.plot(fpr, tpr, label= "Random forest")
plt.plot(fpr1, tpr1, label= "KNN")
plt.plot(fpr2, tpr2, label= "Tree")
plt.plot(fpr3, tpr3, label= "NB")
plt.plot(fpr4, tpr4, label= "Logistic")    
plt.plot(fpr5, tpr5, label= "SVM")
plt.plot(fpr6, tpr6, label= "NeuralNet")
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[335]:


# showing the highest f1 score
a_list = [f1_KNN,f1_random_for,f1_tree,f1_bayes,f1_log_reg,f1_SVC,f1_neu_net]
L = ['KNN','Random Forest','Classification tree','Bayes classification','logistic regression','Support vector','neural network']

max_value = max(a_list)
max_index = a_list.index(max_value)



print("the most efficient method in term of f1_score is %s and the result is %.3f " % (L[max_index], max_value))


# In[ ]:





# In[336]:


#Adaboost
# from sklearn.ensemble import AdaBoostClassifier

# classifier= AdaBoostClassifier()
# parameters = {'n_estimators' : [2000, 5000],
#     'learning_rate' : [0.0001, 0.01, 0.1, 1, 10]}

# ada = hyperp_search(classifier, parameters)


# In[337]:


# adaboost = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 5000)

# fpr7,tpr7=roc(adaboost,X_train,y_train,X_test,y_test)


# # Prediction

# In[392]:


import pickle


# In[393]:


pickle.dump(model_SVC, open('SVC_model.pkl', 'wb'))


# In[394]:


import pickle
import pandas as pd
import numpy as np
import math
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# In[395]:


# load scaler modand el
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
loaded_model = pickle.load(open('SVC_model.pkl', 'rb'))


# In[396]:


df_test = pd.read_csv('tyres_test.csv')


# In[397]:


#data transformation


# In[398]:


df_test.isna().any()


# In[399]:


df_test.drop(columns= 'diameter',inplace = True)


# In[400]:


df_test


# In[401]:


numerical_attributes = ['vulc','perc_nat_rubber','weather','perc_imp','temperature','elevation','perc_exp_comp']
categorical_attributes = ['tread_type','tread_depth','month','wiring_strength','add_layers','tyre_season','tyre_quality']


#categorical

df_cat_1=df_test[['tread_type','tread_depth','month','wiring_strength','add_layers']]

dummies_1 = pd.get_dummies(df_cat_1, columns=['tread_type','tread_depth','month','wiring_strength','add_layers']) 

dummies_1.tail()

print(dummies_1.shape)



#data transformation



#scaling
Numericals_1 = df_test[numerical_attributes]
scaled_df_test = pd.DataFrame(loaded_scaler.transform(Numericals_1))
scaled_df_test.columns = Numericals_1.columns

#concatenate dummies with the other attributes
Col = ['tyre_season','tyre_quality']


# X are the final data that we are going to use 

X1=pd.concat([dummies_1,scaled_df_test,df_test[Col]], axis = 1)



#X is the good dataset for training part but not for data exploration

df_1 = X1

X1.tail()



# In[402]:


y_predictions = loaded_model.predict(X1)


# In[403]:


print(Counter(y_predictions))


# In[404]:


np.savetxt("test_prediction_end.csv", y_predictions.astype(int), delimiter=",", fmt='%.0f')


# In[ ]:





# In[ ]:




