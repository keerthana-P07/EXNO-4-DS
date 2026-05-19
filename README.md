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
      Import Required Libraries
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
```
Feature Scaling using BMI.csv

Load Dataset
```python
df = pd.read_csv('Bmi.csv')  

print("Original Dataset:")
print(df.head())
```

Handle Missing Values
```python
df = df.dropna()
```
Standardscaling

```python
df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])

print("\nStandard Scaled Data:")
print(df_std.head())
```
Minmaxscaling

```python
df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())
```
MaxAbsscaling

```python
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])

print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())
```
Robustscaling

```python
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())
```
Save Scaled Datasets

```python
#df_std.to_csv("BMI_StandardScaled.csv", index=False)
#df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
#df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
#df_robust.to_csv("BMI_RobustScaled.csv", index=False)

print("\nFeature Scaling Completed Successfully.")
```
Import Required Libraries

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
```
Read the CSV file

```python
df = pd.read_csv("income.csv")

print("Dataset Preview:")
print(df.head())
```
Encode Cateorigal Variables

```python
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
                       'relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
```
Encode Target Variable

```python
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes
```
Separate features and Target

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
Scale Data for Chi-Square (Non-negative required)

```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
Filter Method: Chi-Square

```python
selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))
```
Filter Method: ANOVA

```python
selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))
```
Wrapper Method: RFE

```python
logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))
```
Embedded Method: SelectFromModel

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf.fit(X_train, y_train)

selector_embedded = SelectFromModel(rf, threshold="mean")
selector_embedded.fit(X_train, y_train)

selected_features_embedded = X.columns[selector_embedded.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))
```
Accuracy using Embedded Features

```python
X_train_sel = selector_embedded.transform(X_train)
X_test_sel = selector_embedded.transform(X_test)

rf.fit(X_train_sel, y_train)
y_pred = rf.predict(X_test_sel)

print("\nModel Accuracy (Embedded Method):", accuracy_score(y_test, y_pred))
```

# Output:

<img width="461" height="215" alt="Screenshot 2026-03-10 153042" src="https://github.com/user-attachments/assets/4a9d3019-6b25-47c1-b72c-dd4c638a8f95" />


<img width="545" height="246" alt="Screenshot 2026-03-10 153123" src="https://github.com/user-attachments/assets/c6c9c2c6-da04-45df-984c-03eebf8080f6" />

<img width="467" height="227" alt="image" src="https://github.com/user-attachments/assets/4f560c90-4209-45f9-9549-dc6e4f7ca5e2" />

<img width="547" height="224" alt="image" src="https://github.com/user-attachments/assets/c8d1039e-a289-4ea2-9d03-0b08e6381a1c" />

<img width="482" height="235" alt="Screenshot 2026-03-10 153242" src="https://github.com/user-attachments/assets/6fa7f1f3-653d-470c-a991-76ada9823c1c" />

<img width="632" height="285" alt="Screenshot 2026-03-10 153324" src="https://github.com/user-attachments/assets/4b6540a7-bd51-4a7c-af7d-06e8cca9447d" />

<img width="811" height="533" alt="Screenshot 2026-03-10 153356" src="https://github.com/user-attachments/assets/639e780b-4d74-43b6-9bd7-6c4d58c2218e" />

<img width="1096" height="363" alt="Screenshot 2026-03-10 153505" src="https://github.com/user-attachments/assets/cfc0b6e0-cfd3-4a54-8138-7a830dd2c6a4" />

<img width="861" height="312" alt="Screenshot 2026-03-10 153540" src="https://github.com/user-attachments/assets/2fea1a7c-9540-4081-be68-95a8124c080a" />

<img width="1079" height="432" alt="Screenshot 2026-03-10 153637" src="https://github.com/user-attachments/assets/9e8869e2-ac4c-48a0-b197-637586dd44f8" />

<img width="825" height="404" alt="Screenshot 2026-03-10 153817" src="https://github.com/user-attachments/assets/66f978bf-b29d-488c-8bea-4bb0cfffea4f" />

# RESULT:
Thus the Feature Scaling and selection Executed successfully.
