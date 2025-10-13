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
5. Import pandas
   ```python
   import pandas as pd
   ```
   * `pandas` is a Python library for data manipulation and analysis, providing the DataFrame structure.
6. Read the CSV files into DataFrames
   ```python
   df_train = pd.read_csv('train.csv')
   df_test = pd.read_csv('test.csv')
   df_sub = pd.read_csv('sample_submission.csv')
   ```
   * `train.csv` is used for model training, it includes both features and the target `SalePrice`.
   * `test.csv` is used for model prediction, it contains only the features.
   * `sample_submission.csv` shows the expected format for the final submission, containing only `Id` and `SalePrice`.
### Excecuting Programme 
