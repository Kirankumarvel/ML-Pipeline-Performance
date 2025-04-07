
# 🌟 ML Pipeline Performance — Scikit-learn Project with Iris Dataset

Welcome to **ml-pipeline-performance** — a clean and complete machine learning project built using the Iris dataset. This project demonstrates best practices using Scikit-learn Pipelines, cross-validation, GridSearchCV, and visual evaluation with confusion matrices.

---

## 🚀 Project Overview

In this project, we:

- 🔍 **Load and preprocess the Iris dataset**
- 🧪 **Split data with stratification**
- ⚙️ **Create an ML pipeline** using:
  - `StandardScaler` for feature scaling
  - `PCA` for dimensionality reduction
  - `KNeighborsClassifier` for classification
- 🧠 **Train the pipeline** and evaluate performance
- 🔄 **Tune hyperparameters** using `GridSearchCV` with `StratifiedKFold`
- 📈 **Visualize results** using Seaborn heatmaps

---

## 📂 Project Structure

```
ml-pipeline-performance/
├── iris_pipeline.py         # Main script containing the pipeline logic
├── requirements.txt         # Python dependencies
└── README.md                # You're here!
```

---

## 📊 Confusion Matrix

A visual evaluation of model performance using `confusion_matrix` and `seaborn.heatmap`.

![Confusion Matrix](assets/confusion_matrix_example.png) *(Add your saved image here if needed)*

---

## 🧪 How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/Kirankumarvel/ml-pipeline-performance.git
   cd ml-pipeline-performance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**
   ```bash
   python iris_pipeline.py
   ```

---

## 🛠️ Features Used

- `Pipeline` from `sklearn.pipeline`
- `GridSearchCV` with `StratifiedKFold`
- `StandardScaler`, `PCA`, and `KNeighborsClassifier`
- `train_test_split` with `stratify`
- Evaluation: `confusion_matrix`, `.score()`

---

## 🔧 Best Parameters Found

After grid search, the pipeline automatically selects the best combination of:
- Number of principal components
- Optimal `n_neighbors` value for KNN

---

## 📌 Dataset Info

- **Dataset**: Iris Dataset
- **Classes**: Setosa, Versicolor, Virginica
- **Features**: Sepal length, sepal width, petal length, petal width

---

## 📚 Learnings

✅ How to avoid data leakage using pipelines  
✅ Why you should always split **before** preprocessing  
✅ How to do hyperparameter tuning the right way  
✅ Visualizing classification results  

---

## 🤝 Contributions

Feel free to fork the repo, raise issues, or submit PRs to enhance or refactor the code.

---

## 🧠 Bonus Tip

Want to go further? Try swapping in `LogisticRegression`, `SVC`, or `RandomForestClassifier` into the pipeline and rerun the tuning!

---

## 📄 License

This project is licensed under the MIT License.
