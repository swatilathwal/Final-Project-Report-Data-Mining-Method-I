#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[6]:


pip install xlrd


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.metrics import roc_curve


# In[8]:


from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import GradientBoostingClassifier


# In[9]:


data=pd.read_csv("/Users/swatilathwal/Documents/Semester 2/Data Mining/Project/PHY_TRAIN.csv")


# In[10]:


variables=pd.read_excel("/Users/swatilathwal/Documents/Semester 2/Data Mining/Project/Variables.xls")


# In[ ]:





# In[11]:


#data exploration
data


# In[12]:


data.info()


# In[13]:


data.feat78.skew() 


# In[14]:


df=data.describe()
df


# In[15]:


df.to_csv('out.csv')


# In[16]:


import statsmodels.api as sm
#from sklearn.feature_selection
from sklearn.feature_selection import RFE


# In[17]:


''''''''''
df = pd.DataFrame(data, columns = data.columns)
df["target"] = data.target
X = df.drop("target",1)   #Feature Matrix
y = df["target"]          #Target Variable
df.head()
Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues
'''''''''''''


# In[ ]:


''''''''''
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
''''''''''''''''


# In[ ]:


#['feat4', 'feat8', 'feat12', 'feat13', 'feat14', 'feat15', 'feat20', 'feat31', 'feat40', 'feat42', 'feat56', 'feat63', 'feat66', 'feat69', 'feat70', 'feat71', 'feat75']


# In[40]:


imp_feature =['feat4', 'feat8', 'feat12', 'feat13', 'feat14', 'feat15', 'feat20', 'feat31', 'feat40', 'feat42', 'feat56', 'feat63', 'feat66', 'feat69', 'feat70', 'feat71', 'feat75','target']
fdata=data[imp_feature]


# In[41]:


fdata.columns


# In[42]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# In[43]:


skweness_data=data.skew()
skweness_data


# In[44]:


skweness_data.to_csv('skewness.csv')


# In[45]:


#-ve = left skew, +ve= right skew
skweness=fdata.skew() 
skweness.to_csv('skewness_feature.csv')


# In[46]:


skweness


# In[47]:


colum_names= data.columns


# In[48]:


data.isnull().values.any()


# In[49]:


missing_col=data.columns[data.isnull().any()]


# In[50]:


missing_col


# In[51]:


#missing_per=[]
#for i in range (len('missing_col')):
percent_missing = data.feat55.isnull().sum() * 100 / len(data.feat55)
#missing_per.append(percent_missing);
#missing_per
percent_missing


# In[52]:


percent_missing_1= data.feat46.isnull().sum() * 100 / len(data.feat46)
percent_missing_1


# In[53]:


percent_missing_2= data.feat45.isnull().sum() * 100 / len(data.feat45)
percent_missing_2


# In[54]:


percent_missing_3= data.feat44.isnull().sum() * 100 / len(data.feat44)
percent_missing_3


# In[55]:


percent_missing_4= data.feat29.isnull().sum() * 100 / len(data.feat29)
percent_missing_4


# In[56]:


percent_missing_5= data.feat21.isnull().sum() * 100 / len(data.feat21)
percent_missing_5


# In[57]:


percent_missing_6= data.feat22.isnull().sum() * 100 / len(data.feat22)
percent_missing_6


# In[58]:


percent_missing_7= data.feat20.isnull().sum() * 100 / len(data.feat20)
percent_missing_7


# In[59]:


percent_missing_8= data.feat2.isnull().sum() * 100 / len(data.feat2)
percent_missing_8


# In[60]:


missing_indicators = data[missing_col].isnull().astype(int).add_suffix('_M')


# In[61]:


missing_indicators.isnull().any()


# In[62]:


imp = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[63]:


imp.fit(data)
data= imp.transform(data)


# In[64]:


data = pd.DataFrame(data,columns=colum_names)


# In[65]:


data.shape


# In[66]:


missing_indicators.shape


# In[67]:


data.isnull().values.any()


# In[68]:


result = pd.concat([data, missing_indicators], axis=1)


# In[69]:


result


# In[ ]:





# In[70]:


for col in fdata.columns: 
 fdata[col]=fdata[col].fillna(0)


# In[ ]:





# In[71]:


target_col_name = 'target'
feature_target_corr = {}
for col in fdata:
    if target_col_name != col:
        feature_target_corr[col + '_' + target_col_name] = pearsonr(fdata[col], fdata[target_col_name])[0]
print("Feature-Target Correlations")
print(feature_target_corr)


# In[72]:


data_train, data_val = train_test_split(fdata, test_size = 0.3)


# In[73]:


data_train


# In[74]:


x_train=data_train.iloc[:, data_train.columns!='target']
#df.loc[:, df.columns != 'b']


# In[75]:


x_train


# In[76]:


y_train = data_train.target


# In[77]:


y_train


# In[78]:


x_test=data_val.iloc[:, data_train.columns!='target']


# In[79]:


x_test


# In[80]:


y_test=data_val.target


# In[81]:


y_test


# In[82]:


def score(y_pred, y_true):
    error = (np.square(y_pred - y_true)).mean()
    score = 1 - error
    return score

actual_cost = list(data_val['target'])
actual_cost = np.asarray(actual_cost)


# In[83]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[84]:


lasso_coeff=log.coef_


# In[85]:


lasso_coeff


# In[86]:


y_pred01=log.predict(x_test)


# In[87]:


y_pred01


# In[88]:


y_pred=log.predict_proba(x_test)


# In[89]:


y_pred=y_pred[:,1]


# In[90]:


y_pred


# In[91]:


#Logistic Regression accuracy
accuracy_log= score(y_pred, y_test)
baseline=accuracy_log
accuracy_log


# In[92]:


#roc curve without interaction
lr_fpr, lr_tpr, _ = roc_curve(y_test,y_pred)


# In[93]:


lr_fpr


# In[94]:


lr_tpr


# In[95]:


pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()


# In[96]:


lr_auc = roc_auc_score(y_test, y_pred)


# In[97]:


print('Logistic: ROC Area Under the Curve=%.3f' % (lr_auc))


# In[98]:


#Logistic Regression precision
precision=precision_score(y_test, y_pred01, average='binary')


# In[99]:


precision


# In[100]:


con_matrix_lg= confusion_matrix(y_test, y_pred01) 


# In[101]:


con_matrix_lg


# In[102]:


#model2 Logistic regression with Interaction
features=[]
features=x_train.columns


# In[103]:


data_train


# In[104]:


features


# In[105]:


interactions = list()
for f_A in features:
     for f_B in features:
        if f_A > f_B:
            x_train['interaction'] = x_train[f_A] * x_train[f_B]
            x_test['interaction'] = x_test[f_A]* x_test[f_B]
            log.fit(x_train, y_train)
            y_pred=log.predict(x_test)
            acc=score(y_pred, y_test)
            #if acc > baseline:
            print(interactions)
            interactions.append([f_A, f_B, round(acc,4)])
            
            
            


# In[106]:


interactions


# In[107]:


newdf = pd.DataFrame(interactions)


# In[108]:


newdf.sort_values(by=[2], ascending = False)


# In[109]:


x_train['newcol1'] = x_train['feat8'] * x_train['feat71']
x_train['newcol2'] = x_train['feat71'] * x_train['feat31']
x_train['newcol3'] = x_train['feat69'] * x_train['feat56']
x_train['newcol4'] = x_train['feat75'] * x_train['feat15']
x_train['newcol5'] = x_train['feat71'] * x_train['feat66']


# In[110]:


x_test['newcol1'] = x_test['feat8'] * x_test['feat71']
x_test['newcol2'] = x_test['feat71'] * x_test['feat31']
x_test['newcol3'] = x_test['feat69'] * x_test['feat56']
x_test['newcol4'] = x_test['feat75'] * x_test['feat15']
x_test['newcol5'] = x_test['feat71'] * x_test['feat66']


# In[111]:


x_train


# In[112]:


x_train.drop(['interaction'],axis=1, inplace=True)


# In[113]:


x_test.drop(['interaction'],axis=1, inplace=True)


# In[114]:


x_test


# In[115]:


log_interaction = LogisticRegression()
log_interaction.fit(x_train, y_train)


# In[116]:


inter_pred01=log_interaction.predict(x_test)


# In[117]:


inter_pred01


# In[118]:


inter_pred=log_interaction.predict_proba(x_test)


# In[119]:


inter_pred=inter_pred[:,1]


# In[120]:


inter_pred


# In[121]:


#Logistic regression with Interaction accuracy
inter_acc= score(inter_pred, y_test)
inter_acc


# In[122]:


precision_inter=precision_score(y_test, inter_pred01, average='binary')


# In[123]:


#Logistic regression with Interaction precision
precision_inter


# In[124]:


in_fpr, in_tpr, _ = roc_curve(y_test,inter_pred)


# In[125]:


in_tpr


# In[126]:


in_fpr


# In[127]:


pyplot.plot(in_fpr, in_tpr, marker='.', label='Logistic with interaction')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()


# In[129]:


in_auc = roc_auc_score(y_test, y_pred)

print('Logistic: ROC Area Under the Curve=%.3f' % (in_auc))


# In[133]:


con_matrix_in= confusion_matrix(y_test, inter_pred01)


# In[134]:


con_matrix_in


# In[135]:


# model 3 Random Forest


# In[136]:


data


# In[152]:


data.drop(['exampleid'], axis = 1,inplace=True)


# In[138]:


data_trainT, data_valT = train_test_split(data, test_size = 0.3)


# In[139]:


x_trainT = data_trainT.iloc[:, data_trainT.columns!='target']
y_trainT =data_trainT.target
x_testT = data_valT.iloc[:, data_valT.columns!='target']
y_testT= data_valT.target


# In[141]:


y_testT.shape


# In[142]:


RF_Model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[143]:


RF_Model.fit(x_trainT, y_trainT)


# In[144]:


y_testT


# In[145]:


pred= RF_Model.predict_proba(x_testT)


# In[146]:


pred01=RF_Model.predict(x_testT)


# In[147]:


pred01


# In[148]:


pred=pred[:,1]


# In[150]:


pred


# In[151]:


falsepr, truepr, _ = roc_curve(y_testT,pred)


# In[154]:





# In[686]:


truepr


# In[687]:


falsepr


# In[654]:


pyplot.plot(falsepr, truepr, marker='.', label='Random Forest')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()


# In[157]:


RF_auc = roc_auc_score(y_testT,pred)


# In[158]:


print('Random Forest: ROC Area Under the Curve=%.3f' % (RF_auc))


# In[159]:


precisionRF=precision_score(y_testT, pred01, average='binary')


# In[160]:


#model 3 Random Forest precision
precisionRF


# In[161]:


con_matrix_rf= confusion_matrix(y_test,pred01) 


# In[162]:


#model 3 Random Forest matrix
con_matrix_rf


# In[163]:


#model 3 Random Forest accuracy
score(pred,y_testT)


# In[164]:


# Model 4 Gradient Boosting
gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_trainT, y_trainT)


# In[165]:


predGB01= gb.predict(x_testT);


# In[166]:


predGB= gb.predict_proba(x_testT);


# In[167]:


predGB=predGB[:,1]


# In[168]:


predGB


# In[169]:


predGB01


# In[170]:


fp, tp, _ = roc_curve(y_testT,predGB)


# In[171]:


pyplot.plot(fp, tp, marker='.', label='Gradient Boosting')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()


# In[172]:


GB_auc = roc_auc_score(y_testT,predGB)
print('GB: ROC Area Under the Curve=%.3f' % (GB_auc))


# In[670]:


con_matrix_gb= confusion_matrix(y_test,predGB01) 


# In[671]:


#Model 4 Gradient Boosting matrix
con_matrix_gb


# In[672]:


precisionGB=precision_score(y_testT, predGB01, average='binary')


# In[673]:


#Model 4 Gradient Boosting precision
precisionGB


# In[674]:


#Model 4 Gradient Boosting accuracy
score(predGB,y_testT)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




