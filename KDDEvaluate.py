import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from addamm import SimplifiedADDaMM 

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le fichier CSV
def loadDataFromCsv(filePath):
    df = pd.read_csv(filePath, sep=",")
    X = df.drop(columns=['Class']) #, 'Time'
    y = df['Class']
    logger.info(f"Loaded dataset with {len(X)} samples.")
    return X, y

# Appliquer les scalers
def preprocessData(X):
    logger.info("Starting data preprocessing...")
    quantileScaler = QuantileTransformer(output_distribution='normal')
    # standardScaler = StandardScaler()

    # Désactiver le prétraitement pour le moment
    # X = quantileScaler.fit_transform(X)
    # X = standardScaler.fit_transform(X)
    logger.info("Data preprocessing completed.")
    
    return X

# Fonction pour enregistrer la matrice de confusion en tant qu'image
def saveConfusionMatrix(y_true, y_pred, filePath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filePath)
    logger.info(f"Confusion matrix saved as {filePath}")

# Fonction pour tracer les scores de vraisemblance
def plotScores(X, y, scores, filePath):
    plt.figure(figsize=(14, 7))
    inliers_idx = y == 0
    outliers_idx = y == 1

    plt.plot(np.where(inliers_idx)[0], scores[inliers_idx], 'o', label='Inliers')
    plt.plot(np.where(outliers_idx)[0], scores[outliers_idx], 'ro', label='Outliers')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Sample index')
    plt.ylabel('Log Likelihood Score')
    plt.legend()
    plt.title('Log Likelihood Scores')
    plt.savefig(filePath)
    logger.info(f"Log likelihood scores plot saved as {filePath}")

# Évaluation du modèle
def evaluateModel(X, y):
    logger.info("Starting model evaluation...")
    
    # Isoler les outliers et séparer les inliers en ensembles d'entraînement et de test
    outliers_idx = y == 1
    inliers_idx = y == 0
    X_outliers = X[outliers_idx]
    y_outliers = y[outliers_idx]
    X_inliers = X[inliers_idx]
    y_inliers = y[inliers_idx]
    
    # Split les inliers en train et test
    X_train, X_test_inliers, y_train, y_test_inliers = train_test_split(X_inliers, y_inliers, test_size=0.1, random_state=42)
    
    # Ajouter les outliers à l'ensemble de test
    X_test = np.vstack([X_test_inliers, X_outliers])
    y_test = np.hstack([y_test_inliers, y_outliers])
    
    # Initialize and fit the ADDaMM model
    model = SimplifiedADDaMM(bandwidth=0.06)
    model.fit(X_train)
    logger.info("Model fitted, start Testing ...")
    # Make predictions on the test set
    log_prob_test = model.detect(np.array(X_test))
    y_pred_test = np.where(log_prob_test < 0, 1, 0)
    
    # Make predictions on the training set
    log_prob_train = model.detect(X_train)
    y_pred_train = np.where(log_prob_train < 0, 1, 0)
    
    # Save confusion matrix as image
    saveConfusionMatrix(y_test, y_pred_test, 'confusion_matrix_test.png')
    saveConfusionMatrix(y_train, y_pred_train, 'confusion_matrix_train.png')
    
    # Plot log likelihood scores
    plotScores(X_test, y_test, log_prob_test, 'log_likelihood_scores.png')
    
    # Calculate and log metrics for the test set
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    logger.info(f"Test Data - Accuracy: {accuracy}")
    logger.info(f"Test Data - Precision: {precision}")
    logger.info(f"Test Data - Recall: {recall}")
    logger.info(f"Test Data - F1 Score: {f1}")
    
    # Print classification report
    logger.info("Test Data - Classification Report:")
    logger.info(classification_report(y_test, y_pred_test))
    
    # Calculate and log metrics for the training set
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    
    logger.info(f"Train Data - Accuracy: {accuracy_train}")
    logger.info(f"Train Data - Precision: {precision_train}")
    logger.info(f"Train Data - Recall: {recall_train}")
    logger.info(f"Train Data - F1 Score: {f1_train}")
    
    # Print classification report
    logger.info("Train Data - Classification Report:")
    logger.info(classification_report(y_train, y_pred_train))

# Fonction principale
def main(filePath):
    logger.info("Loading data from CSV...")
    X, y = loadDataFromCsv(filePath)
    
    logger.info("Preprocessing data...")
    X_scaled = preprocessData(X)
    
    logger.info("Evaluating model...")
    evaluateModel(X_scaled, y)
    
    logger.info("Script completed successfully.")

# Exécution du script
if __name__ == "__main__":
    filePath = 'datasets/KDD2014_donors_10feat_nomissing_normalised.csv'  # Remplacez par le chemin vers votre fichier CSV
    main(filePath)

