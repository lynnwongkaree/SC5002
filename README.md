# SC5002 Lab Assignment 2
Exploring Linear and Ridge Regression with Cross-Validation 

## Objectives
The goal of this project is to explore the differences between Linear Regression and Ridge Regression by using the dataset [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). We aim to compare the strengths and weaknesses of both models to improve their performance.


## Dataset
The dataset we used is Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). 

The Ames Housing dataset contains detailed information on residential homes in Ames, Iowa. It contains 79 explanatory variables describing almost every aspect of a house and 1460 examples, each representing an individual residential property. This dataset provides a realistic, complex regression problem that combines both numerical and categorical predictors. 


## Steps Taken

### Data Loading
Unzipped the Kaggle dataset and loaded `train.csv` and `test.csv` into pandas Dataframes 


### Data Preprocessing 

All missing numeric values in `df_train` and `df_test` were filled with median of each column from `df_train`. The median is used as it is not as sensitive to outliers compared to the mean. This ensures no missing values remain, avoiding training errors. 

Categorical columns are one-hot encoded using `pd.get_dummies`, which converts text features into numeric form while avoiding multicollinearity. The encoded train and test sets were them aligned to ensure consistent feature columns. The target variable `SalePrice` is then log-transformed to reduce skewness stabilize variance, improving the model's ability to learn linear relationships. 

All features were standardized with `StandardScaler()` so they share the same scale, preventing large-valued variables from dominating the regression model. 


Finally, the dataset was split into training (70%) and validation (30%) sets with `random_state=42` to ensure reproducability, allowing model performance to be evaluated before applying it to the unseen Kaggle test set. 


### Model Training and Evaluation
* Implemented 2 models:
   * Linear Regression
   * Ridge Regression


#### Linear Regression

Model performance was evaluated using 5-fold cross-validation (`cv=5`) with negative mean squared error (MSE) to evaluare model stability. The root mean square error (RMSE) was then calculated for each fold to measure the average prediction error, where lower values indicate a better fit. 

The results were `Linear Regression CV RMSE: [0.12892002 0.28797436 0.16607483 0.11685145 0.21874038]` with a mean of `0.1837122095125966`. 

This suggests that the model's predictions are within about ±20% of the actual house prices performing well for fold (`0.11685145`) but struggling with fold (`0.28797436`). This could indicate that there are maybe outliers or different distributions in that fold. 

The R square score was calculated to measure how much of the variance in the target variable the model could explain. It returns a value of `0.7832441336404377`, meaning the model accounts for approximately 78.3% of the variance in house prices. However, there is still a remaining 21.7% of the variation still unexplained which is likely due to some factors the model did not capture. This includes missing features, noise and nonlinear patterns. 

   
#### Ridge Regression

Used multiple alpha values to test different regularization strengths, to find the optimal balance between bias and variance. Smaller alphas behave like plain Linear Regression, while larger alpha apply stronger penalties. The best alpha was determined automatically using 5-fold cross-validation with MSE as the evaluation metric:
  
```sh
Best alpha for Ridge: 100.0
```

This means that the alpha value 100 achieved the lowest average MSE. The model's performance across folds was then assessed using RMSE, where lower values indicate better predictive accuracy. 

```sh
Ridge CV RMSE scores: [0.12418077 0.25096448 0.1566775  0.11952967 0.17385427]
Ridge CV RMSE mean: 0.16504133909005891
```

This shows that the model's predictions were within about ±18% of the actual house prices, showing consistent and improved performance compared to Linear Regression. The model fits well for fold (`0.11952967`) and struggled with fold (`0.25096448`). This could indicate that there are maybe outliers or different distributions in that fold.

When validated on unseen data, Ridge Regression achieved an R square value of `0.8794372046130793`. This explains aproximately 87.9% of the variance in house prices. However, there is still a remaining 12% of the variation still unexplained which is likely due to some factors the model did not capture. This includes missing features, noise and nonlinear patterns. 


#### Model Comparison
   
Comparing the two R squared values, the R squared value from Ridge Regression is higher than that of Linear Regression. This shows that the Ridge model explains a greater proportion of the variance in house prices and demonstrates generalization better on unseen data. The improvement can be attributed to L2 regularization, which penalizes overly large coefficients, thereby reducing overfitting and stabiliszing the model's predictions without significantly increasing bias. 

#### Ridge Alpha Experiments

To evaluate the effect of regularization strength, multiple alpha values were tested sequentially using 5-fold cross-validation. The code was looped through each alpha, fitting a Ridge Regression model and computing the validation RMSE after reversing the log transformation with `np.expm1()`. 

The results showed that as alpha increased from 0.1 to 100, the validation RMSE steadily decreased from $24,572 to $24,308, indicating improved generalization and reduced overfitting. The model with alpha value 100 achieved the lowest RMSE, making it the optimal regularization strength. However, increasing alpha beyond this point would likely lead to underfitting, as excessive regularization can oversmooth the model and reduce predictive accuracy. 

The final model then used the best alpha value 100 determined earlier, was applied to the unseen test data to generate predictions. The trained `ridge_cv` model first produced predictions in log-transformed form `y_test_pred_log`, which were then converted back to the original dollar scale using `np.expm1()`. This provided the model's final predicted house prices on the test dataset, ready for performance evaluation. 

### Final Model Evaluation and Visualization Phase

#### Data Preprocessing 

Process the data with the same method as shown above. 

#### Correlation heatmap

We computed the correlation matrix of all variables, and the 20 most correlated features with `SalePrice` were selected. A heatmap (Figure 1) was then plotted to visualise how these features relate to one another. This allowed us to identify the impact of different characteristics on house value (`SalePrice`).

<img width="1035" height="800" alt="image" src="https://github.com/user-attachments/assets/702c479d-cd3a-486d-b81b-e77f3d15eef3" />
Figure 1

From Figure 1, factors related to quality (`OverallQual`, `KitchenQual`, `GarageFinish`) and size (`GrLivArea`, `TotalBsmtSF`) exhibit strong positive correlations with price; while factors such materials and finishes (`ExterQual_TA`, `KitchenQual_TA`) have negative or weaker correlations. 

#### Residual distribution 

Residuals (the difference between true and predicted prices on the training data) were plotted. 

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/950e7968-a123-4ac4-a1e8-73410d43cd38" />
Figure 2

Figure 2 shows a slightly right-skewed curve centred close to 0, indicating that the model’s predictions are generally accurate but tend to slightly underestimate higher-priced properties. There are no extreme outliers, meaning that no single variable is disproportionately influencing the predictions.

#### Predicted price distribution 

<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/04fc5feb-ed8f-4175-ad88-0ebc56f9d23e" />
Figure 3

Figure 3 is one of predicted sale prices, and depicts how the model generalises unseen test data. 

The distribution is unimodal and right-skewed, meaning that most properties are “moderately” priced and fewer are very expensive. 
Additionally, there are no visible irregularities - like spikes in the fitted density line or gaps in the distribution. This translates to more stable and realistic predictions within a reasonable range, i.e., the predicted values stay in line with the training data’s actual sale prices. 

#### Feature importance plot 

<img width="1139" height="547" alt="image" src="https://github.com/user-attachments/assets/2b408601-a126-4eb2-a010-cf25f85669c5" />
Figure 3

FIgure 3 shows a barplot that ranks the 20 top features by their absolute coefficient magnitudes. 

From this, we can conclude that more significant factors impacting sale price of house are factors like overall material and quality and above-ground living area (“Overallqual” and “GrLivArea”)- we can conclude that they are primary determinants of higher prices. Conversely, factors like House age (“YearBuilt”)  have comparably smaller coefficients, indicating a smaller influence on price. 

There is also an observable gradual decrease in coefficient sizes. This shows that the model spreads importance more evenly across features and does not rely heavily on one or two. This means that the Ridge Regression prevents overfitting, and is a balanced, reliable model. 



   
