import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

# Load the Sentinel-2 and AVIRIS datasets (preprocessed)
# Use rasterio to open and read remote sensing datasets
def load_raster_data(file):
    with rasterio.open(file) as src:
        return src.read()

sentinel_data = load_raster_data('sentinel_2_composite.tif')
aviris_data = load_raster_data('aviris_data.tif')

# Stack Sentinel-2 and AVIRIS data
combined_data = np.dstack((sentinel_data, aviris_data))

# Load ground truth labels for Kudzu (1) and non-Kudzu (0)
ground_truth_labels = np.load('ground_truth_labels.npy')

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(combined_data.reshape(-1, combined_data.shape[2]),
                                                    ground_truth_labels.ravel(), test_size=0.3)

# 1. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(confusion_matrix(y_test, y_pred_rf))

# 2. Neural Network Classifier
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)

# Predict and evaluate
y_pred_nn = nn.predict(X_test)
print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred_nn)}")
print(confusion_matrix(y_test, y_pred_nn))

# Save the predictions as raster for visualization
def save_raster(data, output_file):
    with rasterio.open(output_file, 'w', driver='GTiff',
                       height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype) as dst:
        dst.write(data, 1)

save_raster(y_pred_rf.reshape(sentinel_data.shape[1], sentinel_data.shape[2]), 'rf_predictions.tif')
save_raster(y_pred_nn.reshape(sentinel_data.shape[1], sentinel_data.shape[2]), 'nn_predictions.tif')
