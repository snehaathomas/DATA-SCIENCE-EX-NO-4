# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")  
print("Original Dataset:")
print(df.head())
```
<img width="362" height="173" alt="image" src="https://github.com/user-attachments/assets/4fb8b9ed-a4d8-48c9-b4c9-649f31c26105" />

```
df = df.dropna()
df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())
```
<img width="451" height="188" alt="image" src="https://github.com/user-attachments/assets/14496937-ff9a-4d25-90fc-1b848d1de837" />

```

df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())
```
<img width="402" height="177" alt="image" src="https://github.com/user-attachments/assets/ebb673da-9eaa-4ea3-ac64-aeaed272e4a2" />

```
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())
```
<img width="402" height="182" alt="image" src="https://github.com/user-attachments/assets/d3cfef9b-ef2f-4d45-8ad8-00b005c9e5ff" />

```
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())
print("\nFeature Scaling Completed Successfully.")
```
<img width="442" height="217" alt="image" src="https://github.com/user-attachments/assets/db9db23e-6cb9-4af0-93db-b1c36ccf0506" />

```
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("income(1) (1).csv")
print("Dataset Preview:")
print(df.head())
```
<img width="782" height="465" alt="image" src="https://github.com/user-attachments/assets/ef02d521-963b-4736-909b-f3a2c108c886" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
                       'relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes
X = df.drop(columns=['SalStat'])
y = df['SalStat']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))

selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))

logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))
```
<img width="1003" height="102" alt="image" src="https://github.com/user-attachments/assets/bb2436c6-bfe2-4ddb-8504-c3921397c55f" />
<img width="902" height="47" alt="image" src="https://github.com/user-attachments/assets/8c3d7687-065f-4917-93fd-4644ac31cfcb" />

```
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
embedded_selector = SelectFromModel(model, threshold='median')
embedded_selector.fit(X, y)
selected_features_embedded = X.columns[embedded_selector.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features_embedded], y,
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy (Embedded Method):", accuracy)
```
<img width="1056" height="95" alt="image" src="https://github.com/user-attachments/assets/cab6894c-2420-4656-9f56-99e627e65d87" />

# RESULT:
Thus the Feature Scaling and selection Executed successfully.
