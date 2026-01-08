
# Data Cleaning & Preprocessing in Python (Complete Practical Cheat Sheet)

This sheet covers **all commonly used commands/functions** from **pandas, NumPy, statistics, SciPy, and sklearn** used in **data cleaning, preprocessing, and analysis practicals**.

---

## 1️⃣ PANDAS – CORE DATA CLEANING (MOST IMPORTANT)

### 🔹 Load & Inspect Data

```python
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.columns
df.index
df.dtypes
```

---

### 🔹 Missing Values Handling

```python
df.isna()
df.isnull()
df.notna()
df.notnull()

df.dropna()
df.dropna(axis=0)      # drop rows
df.dropna(axis=1)      # drop columns
df.dropna(how='all')
df.dropna(thresh=2)

df.fillna(0)
df.fillna(method='ffill')
df.fillna(method='bfill')
df.fillna(df.mean())
```

---

### 🔹 Duplicate Handling

```python
df.duplicated()
df.drop_duplicates()
df.drop_duplicates(subset=['col'])
df.drop_duplicates(keep='first')
```

---

### 🔹 Data Type Conversion

```python
df.astype(int)
df['col'] = df['col'].astype(float)

pd.to_datetime(df['date'])
pd.to_numeric(df['col'], errors='coerce')
```

---

### 🔹 Column & Row Operations

```python
df.rename(columns={'old':'new'})
df.drop('col', axis=1)
df.drop(index=0)

df.sort_values('col')
df.sort_index()
```

---

### 🔹 String Cleaning

```python
df['col'].str.lower()
df['col'].str.upper()
df['col'].str.strip()
df['col'].str.replace('a','b')
df['col'].str.contains('word')
df['col'].str.split(',')
```

---

### 🔹 Filtering & Selection

```python
df[df['age'] > 30]
df.loc[rows, cols]
df.iloc[rows, cols]
df.query('age > 30')
```

---

### 🔹 Grouping & Aggregation

```python
df.groupby('col').mean()
df.groupby('col').sum()
df.groupby('col').agg(['mean','count'])
```

---

### 🔹 Binning (VERY IMPORTANT)

```python
pd.cut(df['age'], bins=[0,18,35,50,100])
pd.qcut(df['salary'], q=4)
```

---

## 2️⃣ NUMPY – NUMERICAL CLEANING

### 🔹 Array Creation & Inspection

```python
np.array()
np.zeros()
np.ones()
np.arange()
np.linspace()
np.shape()
```

---

### 🔹 Missing Values

```python
np.nan
np.isnan(arr)
np.nan_to_num(arr)
```

---

### 🔹 Statistical Functions

```python
np.mean(arr)
np.median(arr)
np.std(arr)
np.var(arr)
np.min(arr)
np.max(arr)
np.percentile(arr, 50)
```

---

### 🔹 Array Cleaning

```python
arr[arr > 0]
np.clip(arr, 0, 100)
np.where(condition, x, y)
```

---

## 3️⃣ PYTHON statistics MODULE (BUILT-IN)

```python
import statistics as stats

stats.mean(data)
stats.median(data)
stats.mode(data)
stats.stdev(data)
stats.variance(data)
```

Used for **basic statistical calculations** (less common than NumPy).

---

## 4️⃣ SCIPY.STATS – ADVANCED STATISTICS

```python
from scipy import stats
```

### 🔹 Z-score & Outliers

```python
stats.zscore(data)
```

### 🔹 Hypothesis Testing

```python
stats.ttest_ind(a, b)
stats.chi2_contingency(table)
stats.f_oneway(a, b, c)
```

### 🔹 Correlation

```python
stats.pearsonr(x, y)
stats.spearmanr(x, y)
```

### 🔹 Distribution Stats

```python
stats.skew(data)
stats.kurtosis(data)
```

---

## 5️⃣ SKLEARN – PREPROCESSING (VERY IMPORTANT)

### 🔹 Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
df['scaled'] = scaler.fit_transform(df[['col']])

scaler = MinMaxScaler()
df['scaled'] = scaler.fit_transform(df[['col']])
```

---

### 🔹 Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded'] = le.fit_transform(df['category'])
```

---

### 🔹 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## 6️⃣ VISUALIZATION (CLEANING SUPPORT)

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
sns.boxplot()
sns.histplot()
sns.countplot()
sns.heatmap()
sns.pairplot()
```

---

## 🔑 FINAL EXAM-READY IMPORT BLOCK

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
```

---

## 🎯 ONE-LINE SUMMARY (VIVA)

> **Data cleaning and preprocessing involve handling missing values, duplicates, outliers, scaling, encoding, and transforming data using pandas, NumPy, SciPy, and sklearn.**
