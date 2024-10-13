import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import xgboost as xgb

# Function to load raster data
def load_raster_data(file):
    with rasterio.open(file) as src:
        return src.read()

# Load Sentinel-2 and AVIRIS datasets (preprocessed as composite images)
sentinel_data = load_raster_data('sentinel_2_composite.tif')
aviris_data = load_raster_data('aviris_data.tif')

# Stack Sentinel-2 and AVIRIS data
combined_data = np.dstack((sentinel_data, aviris_data))

# Load ground truth labels (1 for Kudzu, 0 for non-Kudzu)
ground_truth_labels = np.load('ground_truth_labels.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_data.reshape(-1, combined_data.shape[2]),
                                                    ground_truth_labels.ravel(), test_size=0.3)

### 1. Random Forest (RF) Model ###
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(confusion_matrix(y_test, y_pred_rf))

### 2. Neural Network (NN) Model ###
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print("\nNeural Network Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn)}")
print(confusion_matrix(y_test, y_pred_nn))

### 3. Support Vector Machine (SVM) ###
svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSupport Vector Machine Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(confusion_matrix(y_test, y_pred_svm))

### 4. Naive Bayes (NB) Model ###
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("\nNaive Bayes Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb)}")
print(confusion_matrix(y_test, y_pred_nb))

### 5. Boosted Logistic Regression (XGBoost) ###
blr = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
blr.fit(X_train, y_train)
y_pred_blr = blr.predict(X_test)
print("\nBoosted Logistic Regression (XGBoost) Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_blr)}")
print(confusion_matrix(y_test, y_pred_blr))

### Save predictions as rasters for visualization ###
def save_raster(data, output_file, original_shape):
    data_reshaped = data.reshape(original_shape[1], original_shape[2])
    with rasterio.open(output_file, 'w', driver='GTiff',
                       height=data_reshaped.shape[0], width=data_reshaped.shape[1],
                       count=1, dtype=data.dtype) as dst:
        dst.write(data_reshaped, 1)

# Example: Saving Random Forest predictions
save_raster(y_pred_rf, 'rf_predictions.tif', sentinel_data.shape)

# Similarly, you can save the predictions for the other models:
save_raster(y_pred_nn, 'nn_predictions.tif', sentinel_data.shape)
save_raster(y_pred_svm, 'svm_predictions.tif', sentinel_data.shape)
save_raster(y_pred_nb, 'nb_predictions.tif', sentinel_data.shape)
save_raster(y_pred_blr, 'blr_predictions.tif', sentinel_data.shape)
