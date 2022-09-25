#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # used for linear algebra
import pandas as pd # used for data processing
import matplotlib.pyplot as plt # used for the plot the graph 
import seaborn as sns # used for plot interactive graph
import altair as alt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.linear_model import LinearRegression # for linear regression model
from sklearn.tree import DecisionTreeRegressor # for decision tree regressor
from sklearn.svm import SVR # for support vector regressor
from sklearn.ensemble import GradientBoostingRegressor # for gradient boost regressor
from math import sqrt # for Sqrt
from sklearn.linear_model import LogisticRegression # for logistic regression
from sklearn.metrics import r2_score # for r squared score
from sklearn.metrics import mean_absolute_error # for MAE
from sklearn.metrics import mean_squared_error # for MSE

# reading the all datasets in excel
data=pd.concat(pd.read_excel("C:/Users/konde/Downloads/project/Student_Data_U.xlsx",sheet_name=None,skiprows = 1), 
               ignore_index=True )



#f = pd.ExcelFile('C:/Users/konde/Downloads/project/Student_Data_U.xlsx')
#f
#print(f.sheet_names)

#entered_value=input("enter the sheet name")
#data=f.parse(sheet_name=entered_value,skiprows=1)

#data_std=data
data.head(5)

#In data frame taking CAO Type just standard rows 
data_std=data[data['CAO Type']=='standard']

# finding the unique values in each row
data_std.select_dtypes('object').nunique()

# bbisy-8-y1, bbuss-7-y1
# civl-7-y1, emech-8,

# not progressed 


# In[2]:


#data_std=data
data.head(5)

#In data frame taking CAO Type just standard rows 
data_std=data[data['CAO Type']=='standard']

# finding the unique values in each row
#data_std.select_dtypes('object').nunique()


# finding the percentage of missing values in each column
no_of_columns = data_std.shape[0]
percentage_of_missing_data = data_std.isnull().sum()/no_of_columns
print(percentage_of_missing_data)

data_std.shape

# droping the stage column
data_std.drop("Stage",axis=1,inplace=True);
data_std.drop("Repeat ?",axis=1,inplace=True)

data_std.columns

# removing the na from the dataset
df_std=data_std.dropna(how='all', axis='columns')
df_std
#df_std.describe()

# Remaining the column names
df_std.rename(columns={'Unnamed: 14' : 'sc 01', 'Unnamed: 16' : 'sc 02','Unnamed: 18' : 'sc 03', 'Unnamed: 20' : 'sc 04','Unnamed: 22' : 'sc 05', 'Unnamed: 24' : 'sc 06','Unnamed: 26' : 'sc 07', 'Unnamed: 28' : 'sc 08','Unnamed: 30' : 'sc 09', 'Unnamed: 32' : 'sc 10','Unnamed: 34' : 'sc 11', 'Unnamed: 36' : 'sc 12'}, inplace=True)

#We are finding any missing or null data points of the data set.
df_std.isnull().sum()
df_std.isna().sum()


# filtering the rows which are having the NP, I, W, NONE, X, WH, NA 
bool_vec = (df_std['sc 01'] != 'NP') & (df_std['sc 02'] != 'NP') & (df_std['sc 03'] != 'NP') & (df_std['sc 04'] != 'NP') &(df_std['sc 05'] != 'NP') & (df_std['sc 06'] != 'NP') & (df_std['sc 07'] != 'NP') & (df_std['sc 08'] != 'NP') & (df_std['sc 09'] != 'NP') & (df_std['sc 10'] != 'NP') & (df_std['sc 11'] != 'NP') & (df_std['sc 12'] != 'NP') 
data_std_new=df_std[bool_vec]

bool_vec2=(~(data_std_new['sc 01'].isna())) & (~(data_std_new['sc 02'].isna())) & (~(data_std_new['sc 03'].isna()))& (~(data_std_new['sc 04'].isna())) & (~(data_std_new['sc 05'].isna())) & (~(data_std_new['sc 06'].isna())) & (~(data_std_new['sc 07'].isna())) & (~(data_std_new['sc 08'].isna())) & (~(data_std_new['sc 09'].isna())) &(~(data_std_new['sc 10'].isna())) & (~(data_std_new['sc 11'].isna())) & (~(data_std_new['sc 12'].isna())) 
df2=data_std_new[bool_vec2]

bool_vec3 = (df2['sc 01'] != 'NONE') & (df2['sc 02'] != 'NONE') & (df2['sc 03'] != 'NONE') & (df2['sc 04'] != 'NONE') & (df2['sc 05'] != 'NONE') & (df2['sc 06'] != 'NONE') & (df2['sc 07'] != 'NONE') & (df2['sc 08'] != 'NONE') &  (df2['sc 09'] != 'NONE') & (df2['sc 10'] != 'NONE') & (df2['sc 11'] != 'NONE') & (df2['sc 12'] != 'NONE') 
df3=df2[bool_vec3]

bool_vec4 = (df3['sc 01'] != 'I') & (df3['sc 02'] != 'I') & (df3['sc 03'] != 'I') & (df3['sc 04'] != 'I') & (df3['sc 05'] != 'I') & (df3['sc 06'] != 'I') & (df3['sc 07'] != 'I') & (df3['sc 08'] != 'I') & (df3['sc 09'] != 'I') & (df3['sc 10'] != 'I') & (df3['sc 11'] != 'I') & (df3['sc 12'] != 'I') 
df4=df3[bool_vec4]

bool_vec5 = (df4['sc 01'] != 'W') & (df4['sc 02'] != 'W') & (df4['sc 03'] != 'W') & (df4['sc 04'] != 'W') & (df4['sc 05'] != 'W') & (df4['sc 06'] != 'W') & (df4['sc 07'] != 'W') & (df4['sc 08'] != 'W') & (df4['sc 09'] != 'W') & (df4['sc 10'] != 'W') & (df4['sc 11'] != 'W') & (df4['sc 12'] != 'W') 
df5=df4[bool_vec5]

bool_vec6 = (df5['sc 01'] != 'X') & (df5['sc 02'] != 'X') & (df5['sc 03'] != 'X') & (df5['sc 04'] != 'X') & (df5['sc 05'] != 'X') & (df5['sc 06'] != 'X') & (df5['sc 07'] != 'X') & (df5['sc 08'] != 'X') & (df5['sc 09'] != 'X') & (df5['sc 10'] != 'X') & (df5['sc 11'] != 'X') & (df5['sc 12'] != 'X') 
df6=df5[bool_vec6]

bool_vec7 = (df6['sc 01'] != 'WH') & (df6['sc 02'] != 'WH') & (df6['sc 03'] != 'WH') & (df6['sc 04'] != 'WH') & (df6['sc 05'] != 'WH') & (df6['sc 06'] != 'WH') & (df6['sc 07'] != 'WH') & (df6['sc 08'] != 'WH') & (df6['sc 09'] != 'WH') & (df6['sc 10'] != 'WH') & (df6['sc 11'] != 'WH') & (df6['sc 12'] != 'WH') 
df7=df6[bool_vec7]
df7

df7.sort_values(by=['Term Code'], inplace=True)
df7


# In[3]:


mods = ['module 01','module 02', 'module 03', 'module 04','module 05','module 06','module 07',
        'module 08','module 09', 'Mod 10','Mod 11', 'Mod 12']
scores = ['sc 01','sc 02','sc 03','sc 04','sc 05','sc 06','sc 07','sc 08','sc 09','sc 10','sc 11','sc 12']


# In[4]:


dict_coll=[]
for mod,score in zip(mods,scores):
    res_dict = df7.groupby(mod).groups
    for key in res_dict.keys():
        ind = list(res_dict[key])
        marks = list(df7[score].loc[ind])
        res_dict[key] = marks
    dict_coll.append(res_dict)

for d in dict_coll:
    print(d)


# In[5]:


# Finding the unique subjects
unique_subj = []
for d in dict_coll:
    unique_subj.extend(d.keys())

unique_subj = set(unique_subj)

len(unique_subj)

unique_modules = list(set(unique_subj))

unique_modules


# In[11]:


# divide the dataset from termcode to 40-100% 
df_uptopercentage = df7.iloc[:, :9]

# taken a dataset of modules and score
df_modules = df7.iloc[:, 9:]

df_uptopercentage


# In[12]:



df_modules


# In[7]:


# Automatically assigning the scores for respective modules
mark_dict={}
for module in unique_modules:
    l = []
    for row in range(0,len(df_modules)):
        a = df_modules.iloc[row,:]==module
        flag=False
        for i in range(0,len(a)):
            if a[i]:
                flag=True
                break
        temp = 'NAN'
        if flag==True:        
            temp=df_modules.iloc[row,i+1]
        l.append(temp)
    mark_dict[module] = l

mark_df=pd.DataFrame.from_dict(mark_dict)



#mark_df=pd.read_csv("C:/Users/konde/Downloads/project/Marks_all.csv")
mark_df




#mark_df.to_csv('C:/Users/konde/Downloads/project/Marks_all.csv')


# In[13]:



#mark_df=pd.read_csv("C:/Users/konde/Downloads/project/Marks_all.csv")

#mark_df


# In[15]:


Mark_df_1=mark_df.drop(['Unnamed: 0'], axis=1)


# In[16]:


Mark_df=Mark_df_1.drop(['Unnamed: 0.1'], axis=1)


# In[17]:


Mark_df


# In[18]:


# finding the average score 
mean_list=[]
for i in range(0,len(Mark_df)):
    a=cnt=0
    for j in range(0,len(Mark_df.columns)):
        if Mark_df.iloc[i,j]!='NAN':
            
            a+= float(str(Mark_df.iloc[i,j]))
           
            cnt+=1
    mean_list.append(round(a/cnt,1))

# combining the two dataframes 
df8 = pd.DataFrame(np.hstack([df_uptopercentage,Mark_df]))

# adding the column names 
df8.columns=list(df_uptopercentage.columns) + list(Mark_df.columns)

# inserting the avg score
df8.insert(6,'Avg_score',mean_list)

df8

New_df=df8.dropna()
New_df


# In[20]:


# removing na in a dataframe and assigned a new dataframe name 
New_df=df8.dropna()
New_df

# renaming the dataframe because to avoid the changes after assigning a values for the grades in LC maths and LC english 
new_df=New_df

# adding a colum name called status (how are failed more than one subject they are not progressed)
new_df['Status'] = [0 if s >1 else 1 for s in new_df['0-34%']] 
new_df


# In[21]:



# scatter plot for CAO Pts and AVG Score

alt.data_transformers.disable_max_rows()

cols=input("select which column need to scatterplot with output: ")

fig_scatter = alt.Chart(New_df).mark_point(filled=True).encode(
  y='avg score',x=cols,color='LC Maths')
# making the regression line using transform_regression 
# function and add with the scatter plot
#fig_scatter = fig_scatter + fig_scatter.transform_regression('CAO Pts','avg score').mark_line()
fig_scatter.display()


# In[22]:




#mymap = {'H1':100, 'H2':88, 'H3':77, 'H4':66, 'H5':56, 'H6':46, 'H7':37, 'H8':0, 'O1':56, 'O2':46, 'O3':37, 'O4':28, 'O5':20, 'O6':12, 'O7':0, 'O8':0}

mymap2={'H1':'HA','H2':'HA','H3':'HB','H4':'HB','H5':'HC','O1':'HC','H6':'HD','O2':'HD','H7':'HE','O3':'HE',
        'O4':'HF','O5':'HF','O6':'HG','H8':'HH','O7':'HH','O8':'HH'}
new_df = new_df.applymap(lambda s: mymap2.get(s) if s in mymap2 else s)


# mapping the values for grades

mymap = {'HA':88, 'HB':66, 'HC':56, 'HD':46, 'HE':37, 'HF':20, 'HG':12, 'HH':0}
new_df = new_df.applymap(lambda s: mymap.get(s) if s in mymap else s)



new_df


# In[23]:


# checking for na values in dataframe
new_df.isna().sum()


# In[24]:


# for finding the slope and intercept of maths results
df_M0=new_df[new_df['LC Maths']==0]
df_M12=new_df[new_df['LC Maths']==12]
df_M20=new_df[new_df['LC Maths']==20]
df_M88=new_df[new_df['LC Maths']==88]
df_M56=new_df[new_df['LC Maths']==56]
df_M37=new_df[new_df['LC Maths']==37]
df_M46=new_df[new_df['LC Maths']==46]
df_M66=new_df[new_df['LC Maths']==66]


# In[ ]:





# In[25]:


# for findinng the slope and intercept of english results
df_E0=new_df[new_df['LC English']==0]
df_E12=new_df[new_df['LC English']==12]
df_E20=new_df[new_df['LC English']==20]
df_E37=new_df[new_df['LC English']==37]
df_E46=new_df[new_df['LC English']==46]
df_E56=new_df[new_df['LC English']==56]
df_E66=new_df[new_df['LC English']==66]
df_E88=new_df[new_df['LC English']==88]


# In[ ]:





# In[26]:


# scatter plot CAO Pts Vs Avg score for maths results
print("[H1,H2=88],[H3,H4=66],[H5,O1=56],[H6,O2=46],[H7,O3=37],[O4,O5=20],[O6=12],[O7,O8,H8=0]")
print("")
alt.data_transformers.disable_max_rows()
cols=input("select which column need to scatterplot with output: ")
#making the scatter plot on CAO pts vs Avg score
fig_scatter1 = alt.Chart(new_df).mark_point(filled=True).encode(
  y='avg score',x=cols,color="LC Maths:N")
# making the regression line using transform_regression 
# function and add with the scatter plot
fig_scatter1 = fig_scatter1 + fig_scatter1.transform_regression('CAO Pts','avg score').mark_line().transform_fold(["reg-line"], as_=["Regression", "y"]).encode(alt.Color("Regression:N"))

chart=fig_scatter1.facet(columns=4,facet=alt.Facet('LC Maths',header=alt.Header(labelFontSize=15)))
#final_plot_scatter1.display()
chart.display()

# finding the slope and intercept of regression line for every point 
slope,intercept = np.polyfit(new_df[cols],new_df['avg score'],1)
slope_0,intercept_0 = np.polyfit(df_M0[cols],df_M0['avg score'],1)
slope_12,intercept_12 = np.polyfit(df_M12[cols],df_M12['avg score'],1)
slope_20,intercept_20 = np.polyfit(df_M20[cols],df_M20['avg score'],1)
slope_37,intercept_37 = np.polyfit(df_M37[cols],df_M37['avg score'],1)
slope_46,intercept_46 = np.polyfit(df_M46[cols],df_M46['avg score'],1)
slope_56,intercept_56 = np.polyfit(df_M56[cols],df_M56['avg score'],1)
slope_66,intercept_66 = np.polyfit(df_M66[cols],df_M66['avg score'],1)
slope_88,intercept_88 = np.polyfit(df_M88[cols],df_M88['avg score'],1)


print("slope and intercept of the regression line",cols,"vs average score is: ", slope,intercept)
print("slope and intercept of the regression line",cols,"vs average score of 0: ", slope_0,intercept_0)
print("slope and intercept of the regression line",cols,"vs average score of 12: ", slope_12,intercept_12)
print("slope and intercept of the regression line",cols,"vs average score of 20: ", slope_20,intercept_20)
print("slope and intercept of the regression line",cols,"vs average score of 37: ", slope_37,intercept_37)
print("slope and intercept of the regression line",cols,"vs average score of 46: ", slope_46,intercept_46)
print("slope and intercept of the regression line",cols,"vs average score of 56: ", slope_56,intercept_56)
print("slope and intercept of the regression line",cols,"vs average score of 66: ", slope_66,intercept_66)
print("slope and intercept of the regression line",cols,"vs average score of 88: ", slope_88,intercept_88)

#finding the correlation coefficient and R^2 for each point
list_marks = [0,12,20,37,46,56,66,88]
for i in list_marks:
    temp_r_df = new_df[new_df['LC Maths']==i]
    y = temp_r_df['avg score']
    x = temp_r_df['CAO Pts']

    corr_matrix = np.corrcoef(x, y)
    corr = corr_matrix[0,1]
    R_sq = corr**2

    print("The correlation coefficient and R^2 value of",i,"is ",corr,R_sq)


# In[27]:


# scatter plot CAO Pts Vs Avg score for english results
print("[H1,H2=88],[H3,H4=66],[H5,O1=56],[H6,O2=46],[H7,O3=37],[O4,O5=20],[O6=12],[O7,O8,H8=0]")

print(" ")

alt.data_transformers.disable_max_rows()
cols=input("select which column need to scatterplot with output: ")
#making the scatter plot on CAO pts vs Avg score
fig_scatter2 = alt.Chart(new_df).mark_point(filled=True).encode(
  y='avg score',x=cols,color="LC English:N")
# making the regression line using transform_regression 
# function and add with the scatter plot
fig_scatter2 = fig_scatter2 + fig_scatter2.transform_regression('CAO Pts','avg score').mark_line().transform_fold(["reg-line"], as_=["Regression", "y"]).encode(alt.Color("Regression:N"))

chart=fig_scatter2.facet(columns=4,facet=alt.Facet('LC English',header=alt.Header(labelFontSize=15)))
#final_plot_scatter1.display()
chart.display()

# finding the slope and intercept of regression line for every point 
slope,intercept = np.polyfit(new_df[cols],new_df['avg score'],1)
slope_0,intercept_0 = np.polyfit(df_E0[cols],df_E0['avg score'],1)
slope_12,intercept_12 = np.polyfit(df_E12[cols],df_E12['avg score'],1)
slope_20,intercept_20 = np.polyfit(df_E20[cols],df_E20['avg score'],1)
slope_37,intercept_37 = np.polyfit(df_E37[cols],df_E37['avg score'],1)
slope_46,intercept_46 = np.polyfit(df_E46[cols],df_E46['avg score'],1)
slope_56,intercept_56 = np.polyfit(df_E56[cols],df_E56['avg score'],1)
slope_66,intercept_66 = np.polyfit(df_E66[cols],df_E66['avg score'],1)
slope_88,intercept_88 = np.polyfit(df_E88[cols],df_E88['avg score'],1)

print("slope and intercept of the regression line",cols,"vs average score is: ", slope,intercept)
print("slope and intercept of the regression line",cols,"vs average score of 0: ", slope_0,intercept_0)
print("slope and intercept of the regression line",cols,"vs average score of 12: ", slope_12,intercept_12)
print("slope and intercept of the regression line",cols,"vs average score of 20: ", slope_20,intercept_20)
print("slope and intercept of the regression line",cols,"vs average score of 37: ", slope_37,intercept_37)
print("slope and intercept of the regression line",cols,"vs average score of 46: ", slope_46,intercept_46)
print("slope and intercept of the regression line",cols,"vs average score of 56: ", slope_56,intercept_56)
print("slope and intercept of the regression line",cols,"vs average score of 66: ", slope_66,intercept_66)
print("slope and intercept of the regression line",cols,"vs average score of 88: ", slope_88,intercept_88)


#finding the correlation coefficient and R^2 for each point
list_marks = [0,12,20,37,46,56,66,88]
for i in list_marks:
    temp_r_df = new_df[new_df['LC English']==i]
    y = temp_r_df['avg score']
    x = temp_r_df['CAO Pts']

    corr_matrix = np.corrcoef(x, y)
    corr = corr_matrix[0,1]
    R_sq = corr**2

    print("The correlation coefficient and R^2 value of",i,"is ",corr,R_sq)


# In[28]:


# from dataframe taken columns of maths and english
df10=new_df.iloc[:,[3,4]]
df10

# count plot for score of maths and english
ax=sns.countplot(hue="variable", x="value", data=pd.melt(df10))
#bx=df10.plot.kde()


# In[29]:


# density plot for LC maths and Lc english
sns.kdeplot(data=pd.melt(df10),x='value',hue='variable')


# In[30]:


# for the whole dataset
X, y = new_df.iloc[:, [2,3,4]], new_df.iloc[:, [5]]
print(X)
print(y)


# In[ ]:





# In[31]:


#Model1
# fitting the linear regression model and finding the r^2 of that model
model_1 = LinearRegression()
model_1.fit(X, y)

# predict 
y_pred1 = model_1.predict(X)

# r squared score
r1=r2_score(y,y_pred1)

print(f'The r squared score using Linear Regression is:{r1}')


# In[32]:


#model 2
# fitting the decision tree model and finding the r^2 of that model
model_2 = DecisionTreeRegressor()
model_2.fit(X, y)

# for predict
y_pred2 = model_2.predict(X)

# r squared score for testing
r2=r2_score(y,y_pred2)

print(f'The r squared score for testing using DTR is:{r2}')


# In[33]:


#model 3
# fitting the support vector regressor model and finding the r^2 of that model
model_3 = SVR()
model_3.fit(X,np.ravel(y))

# predict 
y_pred3 = model_3.predict(X)

# r squared score 
r3=r2_score(y,y_pred3)

print(f'The r squared score for testing using SVR is:{r3}')


# In[34]:


from math import sqrt
# fitting the Gradient boosting regressor model and finding the accuracy of that model
model_5 = GradientBoostingRegressor()
model_5.fit(X,np.ravel(y))
# Predict
y_pred5 = model_5.predict(X)
# R- Squared
r5=r2_score(y,y_pred5)

print(f'The r squared score for testing using SVR is:{r5}')


# In[35]:


# regression plot for CAO Pts vs Status
sns.regplot(x=new_df['CAO Pts'], y=new_df['Status'])


# In[36]:


Xs, ys = new_df.iloc[:, [2,3,4]], new_df.iloc[:, [473]]
print(Xs)
print(ys)


# In[37]:


# status p or np
#lbfgs

#model 4
# fitting the support vector regressor model and finding the r^2 of that model
model_4= LogisticRegression(solver='liblinear',random_state=0)
model_4.fit(Xs,np.ravel(ys))

# Predict
Y_pred4 = model_4.predict(Xs)

# r squared score
r4=r2_score(ys,Y_pred4)

print(f'The r squared score using logistic regression is:{r4}')


# In[38]:


# for status
#model 2
# fitting the decision tree model and finding the r^2 of that model
model_2 = DecisionTreeRegressor()
model_2.fit(Xs, ys)

# for predict
Y_pred2 = model_2.predict(Xs)

# r squared score for testing
r2=r2_score(ys,Y_pred2)

print(f'The r squared score for testing using DTR is:{r2}')


# In[39]:


# 0 is Not Progressed
df11=new_df[new_df['Status']==0]
df11


# In[40]:


# 1 is Progressed
df12=new_df[new_df['Status']==1]
df12


# In[41]:


# progressed count of english and maths
df13=df12.iloc[:,[3,4]]
df13

sns.countplot(hue="variable", x="value", data=pd.melt(df13))


# In[43]:


# Not progressed count of maths and english
df14=df11.iloc[:,[3,4]]
sns.countplot(hue="variable", x="value", data=pd.melt(df14))


# In[44]:


# Progressed
x, Y = df12.iloc[:, [2,3,4]], df12.iloc[:, [5]]
print(x)
print(Y)


# In[45]:


# progressed
#model 2
# fitting the decision tree model and finding the r^2 of that model
model_2 = DecisionTreeRegressor()
model_2.fit(x, Y)

# for predict
Y_pred2 = model_2.predict(x)

# r squared score for testing
r2=r2_score(Y,Y_pred2)

print(f'The r squared score for testing using DTR is:{r2}')


# In[46]:


# Not progressed
x1, Y1 = df11.iloc[:, [2,3,4]], df11.iloc[:, [5]]
print(x1)
print(Y1)


# In[47]:


# Not progressed
#model 2
# fitting the decision tree model and finding the r^2 of that model
model_2 = DecisionTreeRegressor()
model_2.fit(x1, Y1)

# for predict
Y1_pred2 = model_2.predict(x1)

# r squared score for testing
r2=r2_score(Y1,Y1_pred2)

print(f'The r squared score for testing using DTR is:{r2}')


# In[ ]:





# In[5]:


z= [10.8,10.4,10.2]


# In[6]:


z1=round(z)
z1


# In[ ]:


#export_graphviz
#pip install pydotplus
#pip install graphviz

from IPython.display import Image  
from sklearn import tree
import pydotplus # installing pyparsing maybe needed

# for black and white decision tree of model
dot_data = tree.export_graphviz(model_2, out_file=None, feature_names = X.columns)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())



# for colour decision tree of model
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(model_2, out_file='tree.dot', 
                feature_names = X.columns,
                class_names = y.columns,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

