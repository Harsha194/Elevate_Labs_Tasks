*****TASK-1*****
# Data Cleaning & Preprocessing - Titanic Dataset
This project demonstrates basic data cleaning and preprocessing for machine learning using the Titanic dataset.
# Data Cleaning & Preprocessing (Titanic Dataset)

## What is Data Preprocessing?
Data preprocessing means turning messy, raw data into clean, organized data that a machine learning model can use.

## Steps Followed
1. **Imported the data** using Pandas.
2. **Explored the data** to understand its structure and missing values.
3. **Handled missing values** by filling in with median (for Age) and mode (for Embarked), and dropped the Cabin column.
4. **Encoded categorical variables** (Sex, Embarked) using one-hot encoding.
5. **Standardized numerical features** (Age, Fare) so they have mean 0 and std 1.
6. **Visualized and removed outliers** in Fare using boxplots and IQR method.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib/Seaborn
- Scikit-learn

- Cleaning and transforming the data helps the model learn better and make more accurate predictions.
- ##Outputs regarding to task
- ![Screenshot 2025-05-26 203829](https://github.com/user-attachments/assets/913dfdf7-ffd6-4f57-8473-17a594b5a933)
- ![Screenshot 2025-05-26 203847](https://github.com/user-attachments/assets/b840dc7d-07e1-4030-b669-caa4df8f38e5)
- ![Screenshot 2025-05-26 203913](https://github.com/user-attachments/assets/98178323-b81f-4082-8846-fd7165b9af28)
- ![Screenshot 2025-05-26 203952](https://github.com/user-attachments/assets/9bf0f0a8-694e-4a35-ad62-39ce4d45bab2)
- ![Screenshot 2025-05-26 204005](https://github.com/user-attachments/assets/404a6840-826b-4961-8b55-5b941bf772f3)

*****TASK-2*****

What is EDA?
EDA is the process of analyzing datasets to summarize their main characteristics, often using visual methods.

Steps Followed
Loaded the dataset using Pandas.

Explored basic statistics (mean, median, std, min, max).

Visualized distributions using histograms and boxplots.

Analyzed relationships between features using pairplots and correlation matrices.

Detected anomalies (outliers, missing values, skewed distributions).

Drew basic inferences about feature importance and data quality.

Tools Used
Python

Pandas (Data manipulation)

Matplotlib/Seaborn (Visualizations)

NumPy (Numerical computations)

Key Outputs
✅ Summary Statistics – Mean, median, and spread of numerical features (Age, Fare, etc.).
📊 Histograms & Boxplots – Visualized distributions and outliers in numeric columns.
🔗 Correlation Matrix – Identified relationships between features (e.g., Pclass vs. Survival).
⚠️ Outlier Detection – Flagged extreme values (e.g., very high Fare values).
📝 Basic Inferences – Noted skewness, missing data, and potential preprocessing steps.

Why EDA Matters
Helps identify data issues (missing values, outliers).

Guides feature engineering (which columns need scaling, encoding, etc.).

Improves model performance by ensuring clean, well-understood data.

- Cleaning and transforming the data helps the model learn better and make more accurate predictions.
- ##Outputs regarding to task
- ![Screenshot 2025-05-27 131458](https://github.com/user-attachments/assets/c76590e7-ebef-452d-9a6d-0957667fd42e)
- ![Screenshot 2025-05-27 131534](https://github.com/user-attachments/assets/bcb530c5-e254-4e12-90aa-538148cd5c36)
- ![Screenshot 2025-05-27 131548](https://github.com/user-attachments/assets/6c48b3a8-d18f-4e0b-81b5-333433ec4714)
- !![111](https://github.com/user-attachments/assets/bf72f826-c464-488f-9107-d21eedd4be17)
- ![Screenshot 2025-05-27 131711](https://github.com/user-attachments/assets/95cdd7eb-b5ea-4644-9a56-815483455614)
- ![Screenshot 2025-05-27 131719](https://github.com/user-attachments/assets/334533ab-0dbf-48fb-990b-79425ced5ecc)



*****TASK-3*****

This project implements Simple and Multiple Linear Regression to predict housing prices using the Housing Price Prediction Dataset. The implementation strictly follows machine learning best practices including data preprocessing, model training, evaluation, and interpretation of results.

📊 Dataset Information
Dataset: Housing.csv
Features:

price: Target variable (in ₹)

area: Total area in sq. ft

bedrooms, bathrooms: Number of rooms

stories: Number of floors

Binary features: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea

furnishingstatus: Categorical (unfurnished, semi-furnished, furnished)

Size: 545 rows × 13 columns

🧰 Requirements
Python 3.8+

Libraries:

bash
pandas
numpy
scikit-learn
matplotlib
seaborn
🚀 Installation & Usage
Clone the repository:

bash
git clone https://github.com/yourusername/housing-price-regression.git
cd housing-price-regression
Install dependencies:

bash
pip install -r requirements.txt
Run the analysis:

bash
python housing_regression.py
🧠 Key Implementation Steps
Data Preprocessing

Convert binary features (yes/no → 1/0)

Label encode furnishingstatus

Train-Test Split (80:20 ratio)

Simple regression: Uses only area feature

Multiple regression: Uses all features

Model Training

LinearRegression from scikit-learn

Separate models for simple and multiple regression

Evaluation Metrics

MAE, MSE, R² for both models

Visualization & Interpretation

Regression line plot

Feature coefficient analysis

📈 Results & Interpretation
Model Performance
Model Type	R² Score	MAE (₹)	Key Insights
Simple Regression	0.28	1,050,000	Limited predictive power using area alone
Multiple Regression	0.67	770,000	2.4x better explanation of price variance
Key Coefficients (Multiple Regression)
Feature	Coefficient (₹)	Interpretation
area	+3,200	Per sq. ft increase
bathrooms	+500,000	Per additional bathroom
airconditioning	+480,000	AC premium
prefarea	+370,000	Preferred location premium
stories	+300,000	Per additional floor

- ![Screenshot 2025-05-29 102700](https://github.com/user-attachments/assets/e581d01d-c2b2-47b3-a524-cdf2cb64a0d3)
- ![Screenshot 2025-05-29 102713](https://github.com/user-attachments/assets/be22773e-8a49-48d0-9a94-9cf7b6452f74)
- ![Screenshot 2025-05-29 102732](https://github.com/user-attachments/assets/6dd6d1c8-214e-43ed-af57-1e77e9ce4ea8)



*****TASK-4*****


Breast Cancer Classification with Logistic Regression
This project demonstrates binary classification using logistic regression on the Breast Cancer Wisconsin Dataset.

Logistic Regression for Binary Classification
What is Logistic Regression?
Logistic regression predicts the probability of a binary outcome (0/1) using the sigmoid function. It's ideal for medical diagnosis tasks like cancer detection.

Steps Followed
Loaded the data using Pandas and explored features

Preprocessed the data by:

Converting diagnosis (M/B) to binary (1/0)

Dropping unnecessary columns (ID, Unnamed: 32)

Split the data into 70% training and 30% testing sets

Standardized features using Scikit-learn's StandardScaler

Trained the model with Logistic Regression

Evaluated performance using:

Confusion matrix

Precision, recall, and F1-score

ROC-AUC score

Tuned the threshold (default 0.5 → tested 0.4)

Visualized results with:

ROC curve

Precision-Recall curve

Sigmoid function

Tools Used
Python

Pandas

NumPy

Matplotlib

Scikit-learn

Key Insights
Achieved 97% accuracy on test data

ROC-AUC score of 0.995 shows excellent class separation

Lowering the threshold to 0.4 increased recall (better for cancer detection)

Visualizations
ROC Curve	Precision-Recall Curve	Sigmoid Function
ROC Curve	Precision-Recall	Sigmoid
"Proper standardization and threshold tuning significantly improve medical diagnosis models."





