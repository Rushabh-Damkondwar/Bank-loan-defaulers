#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import os
import matplotlib.pyplot as plt
import math
pd.set_option('display.max_rows', None) #To display all rows
pd.set_option('display.max_columns', None) # To display all columns
from datetime import date
from datetime import  time
import seaborn as sns


# In[6]:


xyz = pd.read_csv("C:/Users/Rushabh/Downloads/Data science software/python project/XYZCorp_LendingData.txt",encoding = 'utf-8', sep = '\t', low_memory=False)


# In[7]:


#f,ax=plt.subplots(figsize=(9,8))
#sns.heatmap(corr,ax=ax,cmap="YlGnBu",linewidths=0.1)


# ## Lets analyze and understand the data

# In[8]:


xyz.describe()


# In[9]:


print(xyz.dtypes)


# In[10]:


xyz.shape


# # Code to check the unique values function gives negative values or not
#  columns_to_be_removed = ['id','member_id','addr_state','desc','earliest_cr_line','emp_title','pub_rec','pymnt_plan','recoveries','title','zip_code']
# #xyz = xyz.drop(columns_to_be_removed, axis = 1, inplace=True)
# 
# data = [['tom', 10], ['nick', 15], ['juli', 14],['sanky',-10]] 
# df = pd.DataFrame(data, columns = ['Name', 'Age']) 
# 
# df['Age'].unique()
# ##we need to use dataframe['column'] then unique function

# In[11]:


xyz.head()


# In[12]:


xyz['last_credit_pull_d'].head()


# # Analysing Below columns, to get the thorough idea about how sparsed the data is

# ## Column = grade
# XYZ corp. assigned loan grade

# In[13]:


xyz['grade'].unique()


# In[14]:


print(xyz.grade.value_counts())


# ## Column = sub_grade
# XYZ assigned assigned loan subgrade

# In[15]:


xyz['sub_grade'].unique()


# In[16]:


print(xyz.sub_grade.value_counts())


# ## Column = emp_length
# Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 

# In[17]:


xyz['emp_length'].unique()


# In[18]:


xyz['emp_length'].isnull().sum()


# In[19]:


print(xyz.emp_length.value_counts())
#We have 282090 people having more than 10+ years of experience


# ## Column = home_ownership
# 

# In[20]:


xyz['home_ownership'].unique()


# In[21]:


xyz['home_ownership'].isnull().sum()


# In[22]:


print(xyz.home_ownership.value_counts())


# ## Column = verification_status
# 

# In[23]:


xyz['verification_status'].unique()


# In[24]:


print(xyz.verification_status.value_counts())


# ## Column = pymnt_plan
# Indicates if a payment plan has been put in place for the loan

# In[25]:


xyz['pymnt_plan'].unique()


# In[26]:


xyz['pymnt_plan'].isnull().sum()


# ## We`ve got a situation here. Payment plan column has two unique values 'Yes' and 'No'.
# ## Where 'y' has only 5 records, it will definitely impact the accuracy of the model.
# ## We can either remove the specific column or we can do OVER or UNDER SAMPLING.

# In[27]:


print(xyz.pymnt_plan.value_counts())


# ## Column = purpose
# A category provided by the borrower for the loan request. 

# In[28]:


xyz['purpose'].unique()


# In[29]:


print(xyz.purpose.value_counts())


# ## Column = title
# The loan title provided by the borrower

# In[30]:


xyz['title'].unique()


# In[31]:


print(xyz.title.value_counts())


# # Column = delinq_2yrs
# The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years

# In[32]:


xyz['delinq_2yrs'].describe()


# In[33]:


xyz['delinq_2yrs'].unique()


# In[34]:


print(xyz.delinq_2yrs.value_counts())
# We can build a model without converting entire data of this column to 'y' and 'n' that is to '0' and '1'
# We can build a model with converting entire data of this column to 'y' and 'n' that is to '0' and '1' as per solution


# ## Column = application_type
# Indicates whether the loan is an individual application or a joint application with two co-borrowers

# In[35]:


xyz['application_type'].unique()


# ## We`ve got a situation here. Application_type column has two unique values INDIVIDUAL' and 'JOINT'.
# ## Where 'JOINT' has only 442 records, it will definitely impact the accuracy of the model. 

# In[36]:


print(xyz.application_type.value_counts())


# ## Column = policy_code
# Indicates whether the loan is an individual application or a joint application with two co-borrowers

# In[37]:


xyz['policy_code'].unique()


# ### For this policy code column, we have only one type of category for observations.

# In[38]:


xyz.policy_code.value_counts()


# ## Column = acc_now_delinq
# The number of accounts on which the borrower is now delinquent.

# In[39]:


xyz['acc_now_delinq'].unique()


# In[ ]:





# ## 852039 values are zeros, so will hamper the accuracy of the model
# ## It will create class imbalance problem.

# In[40]:


xyz.acc_now_delinq.value_counts()


# # Step 2 - Data Cleaning
# Lets deal with null values

# In[41]:


xyz.isnull().sum()


# In[42]:


null_variables = xyz.isnull().sum(axis=0).sort_values( ascending=False)/float(len(xyz) )
null_variables[ null_variables > 0.75 ]


# ## Let`s remove those columns having more than 75% null values .

# In[43]:


xyz.drop( null_variables[ null_variables > 0.75 ].index, axis = 1, inplace = True ) 
xyz.dropna( axis = 0, thresh = 30, inplace = True )


# ## Let`s deal with the columns, which are less required.
# We will create another model, in which we will see if whether columns like pymnt_plan,application_type, acc_now_delinq hamper the accuracy of model or not

# In[44]:


variables_no_req = ['id', 'member_id','policy_code', 'pymnt_plan',  'application_type', 'acc_now_delinq','title','emp_title', 'zip_code','addr_state','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med']   
#We will create another model, in which we will see if whether columns like pymnt_plan,application_type, acc_now_delinq hamper the accuracy of model or not


# In[45]:


xyz.drop( variables_no_req , axis = 1, inplace = True )


# In[46]:


from sklearn.preprocessing  import LabelEncoder
le = LabelEncoder()


# In[47]:


#xyz.drop('issue_d',axis=1,inplace=True)


# ## Converting the data to numeric format
# ## Column = grade

# In[48]:


xyz['grade'] =le.fit_transform(xyz['grade'])


# In[49]:


xyz['grade'].unique()


# In[50]:


xyz['grade'] = xyz['grade'].astype('category')


# In[51]:


xyz['grade'].describe()


# In[52]:


xyz['grade'].head()


# ## Column = sub_grade

# In[53]:


xyz['sub_grade'] =le.fit_transform(xyz['sub_grade'])


# In[54]:


xyz['sub_grade'].unique()


# In[55]:


xyz['sub_grade'] = xyz['sub_grade'].astype('category')


# In[56]:


xyz['sub_grade'].describe()


# ## Column = purpose

# In[57]:


xyz['purpose'] =le.fit_transform(xyz['purpose'])


# In[58]:


xyz['purpose'].unique()


# In[59]:


xyz['purpose'] = xyz['purpose'].astype('category')


# In[60]:


xyz['purpose'].describe()


# In[61]:


xyz['purpose'].head()


# ## Column = home_ownership

# In[62]:


xyz['home_ownership'] =le.fit_transform(xyz['home_ownership'])


# In[63]:


xyz['home_ownership'].unique()


# In[64]:


xyz['home_ownership'] = xyz['home_ownership'].astype('category')


# In[65]:


xyz['home_ownership'].describe()


# In[66]:


xyz.home_ownership.value_counts()


# ## Column = verification_status

# In[67]:


xyz['verification_status'].unique()


# In[68]:


xyz['verification_status'] =le.fit_transform(xyz['verification_status'])


# In[69]:


xyz['verification_status'].unique()


# In[70]:


xyz['verification_status'] = xyz['verification_status'].astype('category')


# In[71]:


xyz['verification_status'].describe()


# In[72]:


xyz['verification_status'].head()


# In[73]:


#xyz['pymnt_plan'].unique()


# ## Column = initial_list_status

# In[74]:


#xyz['pymnt_plan'] =le.fit_transform(xyz['pymnt_plan'])
#xyz['pymnt_plan'] = xyz['pymnt_plan'].astype('category')
xyz['initial_list_status'].head()


# In[75]:


xyz['initial_list_status'].unique()


# In[76]:


xyz.initial_list_status.value_counts()


# In[77]:


xyz['initial_list_status'] =le.fit_transform(xyz['initial_list_status'])


# In[78]:


xyz['initial_list_status'] = xyz['initial_list_status'].astype('category')


# In[79]:


xyz['initial_list_status'].unique()


# In[80]:


xyz.initial_list_status.value_counts()


# In[81]:


xyz['initial_list_status'].describe()


# In[82]:


xyz['initial_list_status'].head()


# In[83]:


## Column = verification_status_joint


# ## We will use use below code, when we handle class imbalance problem

#  xyz['verification_status_'].unique()
# xyz.verification_status.value_counts()
# xyz['verification_status_joint'] = xyz['verification_status_joint'].fillna('Not Verified')
# xyz['verification_status_joint'] =le.fit_transform(xyz['verification_status_joint'])
# xyz['verification_status_joint'] = xyz['verification_status_joint'].astype('category')

# ## Making some modifications with data, like converting Date and other 'object data types' of python.  

# In[84]:


xyz['term'] = xyz['term'].str.split(' ').str[1]


# In[85]:


xyz['term'].head()


# In[86]:


col_dates = xyz.dtypes[xyz.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    xyz[d] = xyz[d].dt.to_period('M')


# In[87]:


xyz['emp_length'] = xyz['emp_length'].str.extract('(\d+)').astype(float)


# # Filling up those null values of columns with mean,median or mode. 

# In[88]:


xyz.isnull().sum()


# In[89]:


xyz['emp_length'] = xyz['emp_length'].fillna(xyz.emp_length.median())


# In[90]:


xyz['mths_since_last_delinq'].head()


# In[91]:


xyz['mths_since_last_delinq'].describe()


# In[92]:


xyz['mths_since_last_delinq'].median()


# In[93]:


xyz['mths_since_last_delinq'] = xyz['mths_since_last_delinq'].fillna(xyz.mths_since_last_delinq.mean())


# In[ ]:





# In[94]:


#xyz.last_pymnt_d.value_counts()
#Jan-2016 is on the top. 
#Dec-2015 is on number two
#Other dates are not more than 15000 values.


# In[95]:


#xyz['next_pymnt_d'].unique()


# In[96]:


#xyz.next_pymnt_d.value_counts()


# In[97]:


#xyz['next_pymnt_d'] = xyz['next_pymnt_d'].fillna('Mar-2016')


# In[98]:


xyz['tot_coll_amt'].mean()


# In[99]:


xyz['tot_coll_amt'] = xyz['tot_coll_amt'].fillna(xyz.tot_coll_amt.mean())


# In[100]:


xyz['tot_cur_bal'].describe()


# In[101]:


xyz['tot_cur_bal'].mean()


# In[102]:


xyz['tot_cur_bal'] = xyz['tot_cur_bal'].fillna(xyz.tot_cur_bal.mean())


# In[103]:


xyz['total_rev_hi_lim'].unique()


# In[104]:


xyz['total_rev_hi_lim'].describe()


# In[105]:


xyz['total_rev_hi_lim'].mean()


# In[106]:


xyz['total_rev_hi_lim'] = xyz['total_rev_hi_lim'].fillna(xyz.total_rev_hi_lim.median())


# In[107]:


xyz.inq_last_6mths.value_counts()


# In[108]:


xyz = xyz.dropna()


# In[109]:


corr=xyz.corr()


# In[110]:


xyz.shape


# In[111]:


import seaborn as sns


# In[112]:


corr1=xyz.corr()


# In[113]:


f,ax=plt.subplots(figsize=(9,8))
sns.heatmap(corr1,ax=ax,cmap="RdBu",linewidths=0.1)


# In[114]:


xyz.default_ind.value_counts()


# ##  DATA CLEANING DONE 
# 
# 
# ##  FEATURE SELECTION
# ### While performing feature selection operation, do not forget to remove 'Y'-Axis column(Dependent Variable). 

# In[115]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


# # from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# 
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from imblearn.over_sampling import SMOTE
# 
# X_scaled_model_train = preprocessing.scale(xyz_x)
# X_scaled_oot_test = preprocessing.scale(xyz_y)
# print(X_scaled_model_train.shape, X_scaled_oot_test.shape)
# 
# oot_test_months = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']
# xyz_x = xyz.loc [ -xyz.issue_d.isin(oot_test_months) ]
# xyz_y = xyz.loc [ xyz.issue_d.isin(oot_test_months) ]

# In[116]:


#select = SelectFromModel(RandomForestClassifier(n_estimators=100))


# In[117]:


#xyz.head() 


# In[118]:


#xyz_x = xyz.iloc[:, 0:37]
#xyz_y = xyz.iloc[:,-1]


# In[119]:


#from sklearn.model_selection import train_test_split


# In[120]:



#xyz_x_train,xyz_x_test,xyz_y_train,xyz_y_test=train_test_split(xyz_x,xyz_y,test_size=.2,random_state=502)


# In[121]:


#from sklearn.ensemble import RandomForestClassifier
#rfc= RandomForestClassifier()
#rfc.fit(xyz_x_train,xyz_y_train)


# In[122]:


#pred= rfc.predict(xyz_x_test)


# In[123]:


#xyz.shape


# In[124]:


#select.fit(xyz_x,xyz_y)


# In[125]:




#select.transform(xyz_x)


# In[126]:


#select.get_support()


# In[ ]:





# In[127]:


#variable_imp = pd.DataFrame({"Important": list(select.get_support()), "Feature_Name": list(xyz_x.columns)})


# In[128]:


#variable_imp


# # Decision tree for feature selection

# In[129]:


from sklearn.model_selection import  cross_val_score

from sklearn.tree import  DecisionTreeClassifier


# In[130]:


#dtree=DecisionTreeClassifier()


# In[131]:


#dtree.fit(xyz_x,xyz_y)


# In[ ]:





# In[132]:


#score_tree=cross_val_score(dtree,xyz_x,xyz_y,cv=7,scoring='accuracy')


# In[133]:


#score_tree.view()


# In[ ]:





# # Feature selection by using RFE

# In[134]:


#from sklearn.feature_selection import RFE
#from sklearn.svm import  LinearSVC
#select = SelectFromModel(RandomForestClassifier(n_estimators=25000))
#rm=RandomForestClassifier()
#rfe=RFE(rm,15)
#rfe.fit(xyz_x,xyz_y)


# In[ ]:





# In[135]:


#print(rfe.support_)


# In[ ]:





# In[136]:


#FS = pd.DataFrame({"Important":list(rfe.support_), "Feature": list(xyz_x.columns)})


# In[137]:


#FS


# In[ ]:





# In[138]:


features = [
            'loan_amnt', 'funded_amnt_inv', 'term', 'funded_amnt',
            'installment', 'grade','emp_length','out_prncp','out_prncp_inv',
            'home_ownership', 'annual_inc','verification_status','total_pymnt','total_rec_prncp',
            'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths','recoveries','collection_recovery_fee' ,
            'open_acc', 'pub_rec', 'pub_rec','initial_list_status',
            'issue_d','default_ind'
           ]


# In[139]:


xyz = xyz[features]


# In[ ]:





# In[140]:


#features_df=pd.DataFrame(features)


# In[141]:


type(features)


# In[142]:


#features_df.shape


# In[143]:


corr_df=xyz['loan_amnt', 'funded_amnt_inv', 'term', 'funded_amnt',
            'installment', 'grade','emp_length','out_prncp','out_prncp_inv',
            'home_ownership', 'annual_inc','verification_status','total_pymnt','total_rec_prncp',
            'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths','recoveries','collection_recovery_fee' ,
            'open_acc', 'pub_rec', 'pub_rec','initial_list_status',
            'issue_d','default_ind'].corr()


# In[144]:


f,ax=plt.subplots(figsize=(9,8))
sns.heatmap(corr_df,ax=ax,cmap="RdBu",linewidths=0.1)


# In[145]:


xyz.head()


# In[146]:


ot_test_months = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']


# In[147]:


train = xyz.loc [ -xyz.issue_d.isin(ot_test_months) ]
oot_test = xyz.loc [ xyz.issue_d.isin(ot_test_months) ]


# In[148]:


train.shape


# In[149]:


train.head()


# In[150]:


X_model_train = train[train.columns[:-2]]
y_model_train = train['default_ind']
X_oot_test = oot_test[oot_test.columns[:-2]]
y_oot_test = oot_test['default_ind']


# In[151]:


y_oot_test.describe(include='all')


# In[ ]:





# In[ ]:





# In[152]:


#pred


# In[153]:



#from sklearn.metrics import confusion_matrix
#tab1 = confusion_matrix(pred,y_oot_test)


# In[154]:


#tab1


# In[155]:


#tab1.diagonal().sum() / tab1.sum() *100


# # OverSampling

# In[156]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


# In[157]:


X_scaled_model_train = preprocessing.scale(X_model_train)
X_scaled_oot_test = preprocessing.scale(X_oot_test)


# In[158]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled_model_train, y_model_train.values, 
                                                    test_size=0.4, random_state=0)


# In[159]:


X_train, y_train = SMOTE().fit_sample(X_scaled_model_train, y_model_train)
X_test, y_test = X_scaled_oot_test, y_oot_test


# In[160]:


index_split = int(len(X_scaled_model_train)/2)
X_train, y_train = SMOTE().fit_sample(X_scaled_model_train[0:index_split, :], y_model_train[0:index_split])
X_test, y_test = X_scaled_model_train[index_split:], y_model_train[index_split:]


# In[ ]:





# In[161]:


from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#logmodel=LogisticRegression()
#logmodel.fit(X_train,y_train) 

#from sklearn.ensemble import RandomForestClassifier
#rfc= RandomForestClassifier()
#rfc.fit(X_train,y_train)
pred1= dtree.predict(X_test )


# In[162]:


#pred


# In[163]:


from sklearn.metrics import confusion_matrix
tab1 = confusion_matrix(pred1,y_test)
tab1


# In[ ]:


TPR=tp/tp+fp	100%	Precision
TNR=tp/tp+fn	73%	Recall
FPR=fp/fp+tn	4%	
        PREDITED
    0              1
0 207971 (Tn)   506 (Fp)
1 77163 (Fn)    13679 (Tp)


# In[164]:


tab1.diagonal().sum() / tab1.sum() *100


# In[161]:


from sklearn.metrics  import classification_report


# In[162]:


print(classification_report(pred1,y_test))


# In[163]:


#Adaboosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
abc=AdaBoostClassifier(RandomForestClassifier(),n_estimators=250)
abc.fit(X_train,y_train)
pred= abc.predict(X_test)


# In[164]:


from sklearn.metrics import confusion_matrix
tab1 = confusion_matrix(pred,y_test)
tab1


# In[165]:


tab1.diagonal().sum() / tab1.sum() *100


# In[166]:


print(classification_report(pred,y_test))


# In[ ]:





# In[ ]:





# In[207]:


pred_value_log=dtree.predict_proba(X_test)
pred_value_log


# In[208]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
log_roc_auc =roc_auc_score(y_test,pred1)


# In[209]:


log_roc_auc
#just avoid the below code. i have made some changes and getting error. I have to check it first.


# In[195]:


#fpr,tpr,threshold  =roc_curve(y_test,pred_value_log[:,1])


# In[196]:


#fpr,tpr,threshold  =precision_recall_curve(y_test,pred_value_log[:,1])


# In[202]:


precision,recall,threshold  =precision_recall_curve(y_test,pred_value_log[:,1])


# In[206]:


#import matplotlib.pyplot as plt
#plt.plot(fpr,tpr,label="LogModel(Area = %.2f)" % log_roc_auc)
#plt.plot(recall,precision %f1)
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.legend(loc="Upper left")
#plt.title("PROC CURVE")





# In[175]:


#plt.show()


# In[190]:


from sklearn.metrics import confusion_matrix
tab1 = confusion_matrix(pred,y_test)
tab1
#0 and 1
#201316 - 0(NO)
#13682 - 1 (YES)


# In[191]:


tab1.diagonal().sum() / tab1.sum() *100


# In[174]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train) 
pred= logmodel.predict(X_test)


# In[175]:


from sklearn.metrics import confusion_matrix
tab2 = confusion_matrix(pred,y_test  
tab2


# In[ ]:


tab2.diagonal().sum() / tab2.sum() *100


# In[ ]:


y_test.describe(include='all')


# In[ ]:


xyz.default_ind.value_counts()


# In[ ]:





# In[ ]:


from sklearn.svm import SVC
svc_model = SVC()


# In[ ]:


svc_model.fit(X_train,y_train)


# In[ ]:


pred= svc_model.predict(X_test)


# In[ ]:


pred


# In[ ]:


from sklearn.metrics import confusion_matrix
tab1 = confusion_matrix(pred,y_test)
tab1


# In[ ]:


tab1.diagonal().sum() / tab1.sum() *100


# 
#     

# In[ ]:


#from sklearn.ensemble import RandomForestClassifier 
#from sklearn.feature_selection import  SelectFromModel

#select = SelectFromModel(RandomForestClassifier(n_estimators=100))


# In[ ]:





# # Data Visualization is very important before building a model.

# In[ ]:





# In[ ]:





# In[ ]:





# # Feature selection
#  

# In[ ]:





# In[ ]:


#pd.to_datetime(xyz1['date'])
#We can use this code for issu_d column to check


# In[ ]:





# # Model building

# In[ ]:


#from sklearn.model_selection import train_test_split
#xyz_x_train,xyz_x_test,xyz_y_train,xyz_y_test=train_test_split(xyz_x,xyz_y,test_size = .2, random_state = 202)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(xyz_x_train, xyz_y_train)


# In[ ]:


pred= logmodel.predict(xyz_x_test)


# In[ ]:


pred


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


tab2 = confusion_matrix(pred,xyz_y_test)


# In[ ]:


tab2


# In[ ]:


tab2.diagonal().sum() / tab2.sum() *100


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




