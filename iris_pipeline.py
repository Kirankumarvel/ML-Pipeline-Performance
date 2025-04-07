import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Train a model using a pipeline

data = load_iris()
X, y = data.data, data.target
labels = data.target_names

#Instantiate a pipeline consisting of StandardScaler, PCA, and KNeighborsClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Standardize features
    ('pca', PCA(n_components=2, random_state=42)),       # Step 2: Reduce dimensions to 2 using PCA
    ('knn', KNeighborsClassifier(n_neighbors=5))  # Step 3: K-Nearest Neighbors classifier
], memory='cache_directory')

#Task 1. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Fit the pipeline on the training set
pipeline.fit(X_train, y_train)

# Measure and print accuracy on the test data
test_score = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_score:.3f}")

#Get the model predictions
y_pred = pipeline.predict(X_test)
# Removed redundant reassignment of y_pred

#Task 2. Generate the confusion matrix for the KNN model and plot it

# Predict labels for the test set
y_pred = pipeline.predict(X_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Add title and axis labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display the plot
plt.tight_layout()
plt.show()

#Task  3. Describe the errors made by the model.
# make a pipeline without specifying any parameters yet
grid_pipeline = Pipeline(
                    [('scaler', StandardScaler()),
                     ('pca', PCA(random_state=42)),
                     ('knn', KNeighborsClassifier()) 
                    ],
                    memory='cache_directory'
                   )

#Define a model parameter grid to search over
# Hyperparameter grid for PCA and KNN
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 7]
             }  

#Choose a cross validation method
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Task 4. Determine the best parameters

# Set up GridSearchCV
best_model = GridSearchCV(estimator=grid_pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          verbose=2
                         )

#Fit the best GridSearchCV model to the training data
best_model.fit(X_train, y_train)

#Task  5. Evaluate the accuracy of the best model on the test set
test_score = best_model.score(X_test, y_test)
print(f"Best Model Test Accuracy: {test_score:.3f}")

#Display the best parameters
print("Best Parameters:", best_model.best_params_)

#Plot the confusion matrix for the predictions on the test set

# Predict the test set results
y_pred = best_model.predict(X_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Add titles and labels
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()
