# Invasive Species Detection Using Machine Learning and Remote Sensing Data

## Project Overview
This project aims to detect and map the spread of invasive species, specifically Kudzu vine, using machine learning models and remote sensing data. The project leverages **Sentinel-2 multispectral** and **AVIRIS hyperspectral** data to classify and predict Kudzu infestations in a selected geographic area. 

Five machine learning models are implemented:
- Random Forest (RF)
- Neural Network (NN)
- Support Vector Machine (SVM)
- Naive Bayes (NB)
- Boosted Logistic Regression (BLR) using XGBoost

The objective is to compare the performance of these models for classifying Kudzu vs. non-Kudzu areas.

---

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running the Code](#running-the-code)
- [Explanation of Models](#explanation-of-models)
- [Inference and Results](#inference-and-results)
- [Visualization](#visualization)
- [References](#references)

---

## Installation

### **Prerequisites**
- Python 3.6 or higher
- Libraries: rasterio, scikit-learn, tensorflow, xgboost, numpy, pandas

### **Step-by-Step Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/Ebullioscopic/Invasive-Species-Detection.git
   cd invasive-species-detection
   ```

### Dependencies
This project is implemented in Python, and the following libraries are required:

```bash
rasterio==1.2.10
scikit-learn==1.0.2
tensorflow==2.8.0
xgboost==1.5.0
numpy==1.21.5
pandas==1.3.5
```

### Installing via `pip`
Install the dependencies using pip:

```bash
pip install rasterio scikit-learn tensorflow xgboost numpy pandas
```

Alternatively, you can install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Colab Usage
If you are using Google Colab, you can run the following to install the necessary packages:

```bash
!pip install rasterio scikit-learn tensorflow xgboost numpy pandas
```

---

## Dataset Preparation

### Remote Sensing Data
The project uses two primary datasets:
- **Sentinel-2 Multispectral Data**: Download from [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home).
- **AVIRIS Hyperspectral Data**: Download from [NASA's AVIRIS portal](https://aviris.jpl.nasa.gov/dataportal/).

Both datasets should be preprocessed to create composite images for analysis:
1. Sentinel-2: Create seasonal composites (Spring, Summer, Autumn, Winter) and select bands 2, 3, 4, 5, 6, 7, 8, 8a, 11, 12.
2. AVIRIS: Ensure that all 224 bands are included after resampling to match the Sentinel-2 spatial resolution.

### Ground Truth Labels
Ground truth labels for Kudzu and non-Kudzu areas are necessary for training the machine learning models. You can create this manually or use external sources like the **EDDMapS dataset**.

The ground truth labels should be stored in a `.npy` file, where `1` represents Kudzu and `0` represents non-Kudzu.

---

## Project Structure
```
|-- main.py                   # Main execution file for training and evaluation
|-- readme.md                 # Project README file
|-- requirements.txt           # Dependencies list
|-- sentinel_2_composite.tif    # Preprocessed Sentinel-2 composite (input data)
|-- aviris_data.tif            # Preprocessed AVIRIS hyperspectral data (input data)
|-- ground_truth_labels.npy    # Ground truth labels for classification
|-- output/                    # Directory for saving model predictions
|-- data/                      # Directory for storing input datasets
```

---

## Usage

### Running the Code

1. **Prepare the Data**: Ensure that Sentinel-2, AVIRIS, and ground truth labels are available in the appropriate format.
2. **Run the Project**:
   Run the `main.py` script to train and evaluate all models:

   ```bash
   python main.py
   ```

   This will:
   - Load the Sentinel-2 and AVIRIS data.
   - Train Random Forest, Neural Network, SVM, Naive Bayes, and Boosted Logistic Regression models.
   - Evaluate each model's accuracy and confusion matrix on the test set.
   - Save the predictions as GeoTIFF raster files for visualization.

---

## Explanation of Models

The following machine learning models are used in the project:

1. **Random Forest (RF)**:
   - A decision tree-based ensemble model that aggregates multiple trees to improve accuracy and reduce overfitting.
   - Configured with 1000 trees.

2. **Neural Network (NN)**:
   - A feed-forward neural network with one hidden layer for classification.
   - Configured with 100 hidden units and a maximum of 1000 training iterations.

3. **Support Vector Machine (SVM)**:
   - A popular machine learning classifier using a radial basis function (RBF) kernel to separate Kudzu from non-Kudzu areas.

4. **Naive Bayes (NB)**:
   - A probabilistic model based on Bayes’ theorem, assuming features are conditionally independent.

5. **Boosted Logistic Regression (BLR)** (using XGBoost):
   - An ensemble model based on logistic regression boosted using gradient boosting trees, implemented using XGBoost.

### Hyperparameters
- Random search or grid search is recommended for tuning hyperparameters.
- In this implementation, default parameters are used, but customization is possible for better results.

---

## Inference and Results

### Accuracy and Confusion Matrix
Each model will print its **accuracy** and **confusion matrix** after evaluation. Here’s an example of Random Forest output:

```
Random Forest Results
Accuracy: 0.91
Confusion Matrix:
[[500  10]
 [ 15 100]]
```

The confusion matrix shows how well the model classifies Kudzu (`1`) and non-Kudzu (`0`).

### Inferences
- **Random Forest** and **Neural Network** tend to perform well on this task with high accuracy.
- **SVM** and **XGBoost** also show good results, while **Naive Bayes** tends to underperform due to its simplicity.

---

## Visualization

### Saving Predictions
The predictions from each model are saved as GeoTIFF raster files for visualization. These files can be viewed in GIS software such as QGIS.

Example saved prediction:
```
rf_predictions.tif  # Random Forest classification result
```

To visualize:
- Open the `.tif` file in QGIS or similar software to see the classified areas of Kudzu (1) and non-Kudzu (0).

### Saving the Results (in `main.py`)
```python
save_raster(y_pred_rf, 'output/rf_predictions.tif', sentinel_data.shape)
```

---

## References

1. Tobias Jensen, Frederik Seerup Hass, Mohammad Seam Akbar, Philip Holm Petersen, Jamal Jokar Arsanjani. **Employing Machine Learning for Detection of Invasive Species using Sentinel-2 and AVIRIS Data: The Case of Kudzu in the United States**. Sustainability 2020.
2. [Sentinel-2 Data](https://scihub.copernicus.eu/dhus/#/home) from Copernicus Open Access Hub.
3. [AVIRIS Hyperspectral Data](https://aviris.jpl.nasa.gov/dataportal/) from NASA’s AVIRIS Data Portal.
4. Python Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
5. XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)