# House Price Predictor

## Overview

This project builds a simple machine learning model to predict house prices using linear regression.  
The workflow includes:

- Loading and cleaning housing data
- Encoding categorical features
- Feature normalization
- Train/test data splitting
- Training a linear regression model
- Evaluating model performance

The project is implemented in a Jupyter Notebook using Python and scikit-learn.

---

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- matplotlib
- scikit-learn

---

## Project Workflow

### 1. Import Libraries

The notebook imports the required libraries for:

- Data processing
- Visualization
- Numerical computation
- Machine learning

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
```

---

### 2. Load Dataset

The dataset is loaded from:

```python
housePrice.csv
```

```python
df = pd.read_csv("housePrice.csv")
```

---

### 3. Data Cleaning

Missing values in the `Address` column are removed:

```python
df['Address'].replace('', np.nan, inplace=True)
df.dropna(subset=['Address'], inplace=True)
```

---

### 4. Encode Categorical Data

The `Address` column is converted into numerical values using `LabelEncoder`.

```python
le = preprocessing.LabelEncoder()
le.fit(df["Address"])
df["address_number"] = le.transform(df["Address"])
```

---

### 5. Feature Processing

- Area values are cleaned by removing commas
- Area is converted to float
- Extremely large areas are filtered out
- Boolean values are converted to integers

```python
df['Area'] = df['Area'].str.replace(',', '')
df['Area'] = df['Area'].astype(float)
df = df[df['Area'] < 2000]
```

Boolean features:

- Parking
- Warehouse
- Elevator

are converted from `True/False` to `1/0`.

---

### 6. Normalization

The `Area` feature is normalized:

```python
df["Area"] = df["Area"] / max(df["Area"])
```

---

### 7. Train/Test Split

The dataset is split into:

- 80% training data
- 20% testing data

```python
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
```

---

### 8. Model Training

A linear regression model is trained using:

Features:
- Area
- Room count
- Parking
- Warehouse
- Elevator
- Encoded address

Target:
- Price

```python
regr = linear_model.LinearRegression()
regr.fit(x, y)
```

---

### 9. Model Evaluation

The model performance is evaluated using the variance score (`R² score`).

```python
print('Variance score: %.2f' % regr.score(x, y))
```

---

## Features Used

| Feature | Description |
|---|---|
| Area | Property size |
| Room | Number of rooms |
| Parking | Parking availability |
| Warehouse | Warehouse availability |
| Elevator | Elevator availability |
| address_number | Encoded property address |

---

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Open the notebook:

```bash
jupyter notebook maktab.ipynb
```

3. Run all notebook cells.

