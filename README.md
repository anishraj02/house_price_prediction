🏠 House Price Prediction System (End-to-End Machine Learning Project)

An end-to-end regression-based machine learning system to predict house prices using the Ames Housing Dataset.
The project covers data analysis, feature engineering, model development, deployment, and a real-time interactive web application.

📌 Project Highlights

• Comprehensive Exploratory Data Analysis (EDA) with business insights

• Robust preprocessing using Scikit-learn Pipelines & ColumnTransformer

• Trained and compared multiple regression models

• Hyperparameter tuning for optimal performance

• Feature importance for interpretability

• Deployed real-time prediction system using Streamlit

📊 Dataset

Ames Housing Dataset
Contains 79 features describing residential properties including:

• Property quality & size

• Basement and garage details

• Neighborhood (location)

• Construction year & renovations

Target variable: SalePrice

🧠 Machine Learning Workflow
1️⃣ Data Analysis

• Missing value handling

• Skewness treatment using log transformation

• Outlier detection

• Correlation analysis

2️⃣ Feature Engineering

• Numerical scaling

• Categorical encoding using OneHotEncoder

• Automated preprocessing pipeline

3️⃣ Model Training

Models evaluated:
• Linear Regression

• Random Forest Regressor

• Gradient Boosting Regressor

4️⃣ Best Model

✔ Tuned Gradient Boosting Regressor
✔ R² ≈ 0.93
✔ RMSE ≈ $20,000

📈 Feature Importance (Top Drivers)

• Overall Quality

• Living Area (GrLivArea)

• Basement Size

• Garage Area

• Construction Year

These align with real-world real estate pricing factors.

🌍 Web Application (Streamlit)
Features:

• Quick price estimation

• Location-aware predictions

• User-friendly interface

• Real-time ML inference

🛠 Tech Stack

• Python

• Pandas, NumPy

• Scikit-learn

• Matplotlib, Seaborn

• Streamlit

• Joblib

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

📂 Project Structure
DS_HousePP/
│
├── app.py
├── HousePricePred.ipynb
├── house_price_pipeline.joblib
├── data.csv
├── requirements.txt
└── README.md

📌 Future Improvements

• Add model explainability with SHAP

• Integrate map-based location visualization

• Extend to price trend forecasting

• Cloud deployment with CI/CD

👤 Author

K B Anishraj
Aspiring Data Scientist | Machine Learning Enthusiast
