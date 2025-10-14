# SC5002 Lab Assignment 2
Exploring Linear and Ridge Regression with Cross-Validation 

## Objectives
The goal of this project is to explore the differences between Linear Regression and Ridge Regression by using the dataset [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). We aim to improve the performance of both models by analyzing the home sale price prediction. 


## Dataset
The dataset we used is Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). 

The Ames Housing dataset contains detailed information on residential homes in Ames, Iowa. It contains 79 explanatory variables describing almost every aspect of a house, 1460 training examples and 1459 test examples , each representing an individual residential property. This dataset provides a realistic, complex regression problem that combines both numerical and categorical predictors. 

### Key Details 
(do we want to add more?)


## Getting Started
### Prerequisites

### Installation
1. Install the Kaggle Python Package from PyPI.
   ```sh
   pip install kaggle
   ```
   
2. Set Kaggle API credentials
   ```python
   import os
   os.environ['KAGGLE_USERNAME'] = 'your_kaggle-username'
   os.environ['KAGGLE_KEY'] = 'your_kaggle-api-key'
   ```
   
3. Download the dataset
   ```sh
   !kaggle competitions download -c house-prices-advanced-regression-techniques
   ```
    * This command downloads a ZIP file:
      ```python
      house-prices-advanced-regression-techniques.zip
      ```
      
4. Open and extract the ZIP file
   ```python
   import zipfile
    with zipfile.ZipFile('house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    zip_ref.extractall()
   ```
   * `ZipFile('filename.zip','r')` opens the ZIP file in read mode.
   * `extractall()` extracts all files inside the ZIP to the current working directory.
   * This creates files like:
     ```sh
     train.csv
     test.csv
     sample_submission.csv
     data_description.txt
     ```
     
5. Read the CSV files into DataFrames
   ```python
   df_train = pd.read_csv('train.csv')
   df_test = pd.read_csv('test.csv')
   df_sub = pd.read_csv('sample_submission.csv')
   ```
   * `train.csv` is used for model training, it includes both features and the target `SalePrice`.
   * `test.csv` is used for model prediction, it contains only the features.
   * `sample_submission.csv` shows the expected format for the final submission, containing only `Id` and `SalePrice`.
  
6. Preview the first few rows of each DataFrame
   ```python
   print("Train head:")
   print(df_train.head())
   
   print("\nTest head:")
   print(df_test.head())
   
   print("\nSample Submission head:")
   print(df_sub.head())
   ```
   *`.head()` prints the first 5 rows
   *This allows us to check if the files were loaded correctly.
   
### Data Preprocessing 
1. Handling Missing Values
   ```python
   num_cols = df_train.select_dtypes(include=['int64','float64']).columns.tolist()
   ```
   * Select columns in `df_train` whose data types are either integers or decimals (numeric) and convert them into a Python list
   ```python
   num_cols.remove('SalePrice')  # exclude target
   ```
   * Remove `SalePrice` as it is not a feature, it's the target.
   ```python
   for col in num_cols:
       median = df_train[col].median()
       df_train[col] = df_train[col].fillna(median)
       df_test[col] = df_test[col].fillna(median)
   ```
   * For each selected numeric columns in `df_train`, find the median and fill missing values in `df_train` and `df_test` with it.
   * The median value is used as it is not as sensitive to outliers compared to the mean.
   * This ensures that there will be no NaN values remaining, avoiding errors during the model training, ensuring all samples are valid and complete for modelling.
   
2. Encode Categorical Variables
   ```python
   cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()!
   ```
   * Select columns in `df_train` with categorical columns (string) and convert them into a Python list.
   ```python
   df_train_encoded = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
   df_test_encoded = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
   ```
   * Use `pd.get_dummies` (One-Hot Encoding) to convert categorical columns into binary (0/1) columns in both `df_train` and `df_test`.
   * Use `drop_first=True` to avoid multicollinearity.
   * This turns text categories into numbers the model can understand.

3. Align Train and Test Columns
   ```python
   df_test_encoded = df_test_encoded.reindex(columns=df_train_encoded.columns.drop('SalePrice'), fill_value=0)
   ```
   * `reindex()` aligns `df_test` columns with `df_train`.
   * `fill_value=0` fills missing columns with 0.
   * This ensures both datasets have identical feature structures for prediction.

4. Seperate Features and Target
   ```python
   X = df_train_encoded.drop('SalePrice', axis=1)
   y = np.log1p(df_train_encoded['SalePrice']) # log transform
   ```
   * Split the data into:
      * `X`: All input features.
      * `y`: `SalePrice` only.
   * `drop('SalePrice', axis=1)` removes the whole column `SalePrice` from `df_train`.
   * `np.log1p()` log-transforms `SalePrice` to reduce skewness and make the data more normally distributed.
   * This improves regression performance by stabilizing variance.

5. Scale Numerical Features
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   X_test_scaled = scaler.transform(df_test_encoded)
   ```
   * `StandardScaler()` standardizes the data such that each column has mean = 0 and standard deviation = 1
   * `fit()` learns the mean and standard deviation from training data to establish scaling parameters.
   * `transform()` uses the parameters determined from `fit()` to scale data.
   * Only `transform()` is used on the test set as the same mean and standard deviation has to be used to ensure consistant scaling of the test set with the training set. 
   * This normalizes features so all are on the same scale.

6. Train-test Split for Evaluation
   ```python
   X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
   ```
   * Splits data into 70/30:
      * 70% training for fitting the model
      * 30% for validation. testing the model
   * `random_state=42` ensures reproducability.
   * This allows the evaluation of the model performance before using the actual Kaggle test set. 


### Model Training and Evaluation
1. Linear Regression with k-fold CV
   ```python
   lin_reg = LinearRegression()
   ```
   * This creates an ordinary least square model.

   ```python
   lin_cv_scores = cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
   ```
   * `cv=5` splits the data into 5 folds where each fold is used once as a hold-out.
   * `cross_val_scores` returns negative mean squared error, where a higher number means better performance. 

