# House-Price-Prediction-Model

## 📌 Project Overview
This project builds a **Machine Learning model** to predict house prices based on various features such as location, number of bedrooms, and square footage. We compare two models: **Linear Regression** and **Random Forest Regressor** to evaluate performance and accuracy.

## 📂 Dataset Overview
- The dataset consists of **4600 rows and 18 columns**.
- The **target variable** is `price`.
- Features include **numerical** (e.g., `sqft_living`, `bathrooms`) and **categorical** (e.g., `city`, `statezip`).

## 🔧 Technologies Used
- **Programming Language**: Python 🐍
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

## 🚀 Steps Implemented

### 1 Data Preprocessing
- **Dropped unnecessary columns**: `date`, `street`, and `country` (as they don’t contribute to price prediction).
- **Handled missing values**: No missing records were found.
- **Encoded categorical variables**: Used **One-Hot Encoding** to convert categorical data (`city`, `statezip`) into numerical form.

### 2 Exploratory Data Analysis (EDA)
- **Plotted House Price Distribution**: To check for skewness in house prices.
- **Generated a Correlation Heatmap**: To analyze relationships between features and the target variable (`price`).

### 3 Model Training
We trained **two models** to predict house prices:
####  Linear Regression
- A simple and interpretable model that assumes a linear relationship between features and the target variable.
- Used as a **baseline model** to compare with more complex models.

####  Random Forest Regressor
- A **powerful ensemble model** that builds multiple decision trees and averages the predictions.
- Captures **non-linear relationships** better than Linear Regression.
- Uses **100 trees** to balance performance and training time.

### 4 Model Evaluation
We evaluated both models using:
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted prices.
- **Mean Squared Error (MSE)**: Penalizes large errors more than MAE.
- **R² Score**: Measures how well the model explains variability in house prices (higher is better).

## 📊 Results & Findings
| Model                 |   MAE($) |      MSE($)   | R² Score |
|-----------------------|----------|---------------|----------|
| **Linear Regression** | 53457.82 | 7401323588.68 | 0.85062  |
| **Random Forest**     | 7082.10  | 320744740.29  | 0.99352  |


## 💡 Future Improvements
- **Hyperparameter Tuning**: Use of **GridSearchCV** to optimize Random Forest parameters.
- **Trying Other Models**: Experiment with Gradient Boosting, XGBoost, and Neural Networks.

## 🛠 How to Run the Project
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```
2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Run the Python script**:
   ```bash
   python house_price_prediction.py
   ```
4. **Compare model performance** and fine-tune as needed.

## ✨ Conclusion
- **Linear Regression** is a good starting point but lacks complexity.
- **Random Forest** significantly improves accuracy by handling **non-linearity and interactions**.
- Future optimizations can further **enhance prediction accuracy**.

---

