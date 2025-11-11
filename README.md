# 🎯 Gender Classification using Machine Learning Algorithms

This project predicts gender (Male/Female) based on physical characteristics using various **Machine Learning algorithms**.  
It demonstrates data preprocessing, model building, and performance evaluation using eight classification models.


## 📘 Project Overview

The goal of this project is to build a predictive model that classifies gender based on measurable features like hair length, forehead size, and facial structure indicators.
The project includes:
- Data preprocessing and feature encoding  
- Exploratory Data Analysis (EDA)  
- Implementation of 8 different ML algorithms  
- Model evaluation and comparison using multiple performance metrics  


## 📂 Dataset: `gender_classification.csv`

The dataset contains structured data describing physical attributes of individuals, along with their gender labels.

### 🧩 Dataset Columns

| Feature | Description |
|----------|-------------|
| `long_hair` | 1 if the person has long hair, else 0 |
| `forehead_width_cm` | Forehead width in centimeters |
| `forehead_height_cm` | Forehead height in centimeters |
| `nose_wide` | 1 if the person has a wide nose, else 0 |
| `nose_long` | 1 if the person has a long nose, else 0 |
| `lips_thin` | 1 if the person has thin lips, else 0 |
| `distance_nose_to_lip_long` | 1 if the distance between nose and lips is long, else 0 |
| `gender` | Target variable — 1 for Male, 0 for Female |

🧠 **Type:** Supervised Learning (Classification)  
📄 **Labels:** Binary (Male/Female)  
📈 **Data Count:** ~1000 samples  


## 🤖 Algorithms Used

The following **Machine Learning algorithms** were implemented and compared:

1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Decision Tree  
4. Random Forest  
5. Support Vector Machine (SVM)  
6. Naive Bayes  
7. Gradient Boosting Classifier  
8. XGBoost Classifier

Each model was evaluated on accuracy, precision, recall, and F1-score metrics.


## 🧰 Technologies Used
- **Python**
- **NumPy** and **Pandas** – for data manipulation
- **Matplotlib** and **Seaborn** – for visualization
- **Scikit-learn** – for model training and evaluation
- **Jupyter Notebook**

## 🏁 **Conclusion**

This project used eight machine learning algorithms to classify gender based on physical traits from the `gender_classification.csv` dataset.
Among all models, the **SVM (RBF kernel)** gave the best result with **97% accuracy**, followed by Logistic Regression, KNN, Naive Bayes, and AdaBoost with accuracies above **>=96%**.
Overall, the results show that properly scaled traditional ML models can achieve excellent accuracy for gender classification.



