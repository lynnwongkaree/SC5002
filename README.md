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
   * Use `pd.get_dummies` (One-Hot Encoding) to convert categorical columns into binary (0/1) columns.
   * 
