import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from addamm import SimplifiedADDaMM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset from CSV
def loadDataFromCsv(filePath):
    df = pd.read_csv(filePath, sep=",")
    X = df.drop(columns=['Class'])  # Drop target column
    y = df['Class']
    logger.info(f"Loaded dataset with {len(X)} samples.")
    return X, y

# Apply scalers to the dataset
def preprocessData(X):
    logger.info("Starting data preprocessing...")
    quantileScaler = QuantileTransformer(output_distribution='normal')
    
    # Optionally preprocess the data
    # X = quantileScaler.fit_transform(X)
    logger.info("Data preprocessing completed.")
    
    return X

# Save confusion matrix as an image
def saveConfusionMatrix(yTrue, yPred, filePath):
    cm = confusion_matrix(yTrue, yPred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filePath)
    logger.info(f"Confusion matrix saved as {filePath}")

# Plot likelihood scores
def plotScores(X, y, scores, filePath):
    plt.figure(figsize=(14, 7))
    inliersIdx = y == 0
    outliersIdx = y == 1

    plt.plot(np.where(inliersIdx)[0], scores[inliersIdx], 'o', label='Inliers')
    plt.plot(np.where(outliersIdx)[0], scores[outliersIdx], 'ro', label='Outliers')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Sample index')
    plt.ylabel('Log Likelihood Score')
    plt.legend()
    plt.title('Log Likelihood Scores')
    plt.savefig(filePath)
    logger.info(f"Log likelihood scores plot saved as {filePath}")

# Evaluate the model performance
def evaluateModel(X, y):
    logger.info("Starting model evaluation...")
    
    # Separate inliers and outliers
    outliersIdx = y == 1
    inliersIdx = y == 0
    XOutliers = X[outliersIdx]
    yOutliers = y[outliersIdx]
    XInliers = X[inliersIdx]
    yInliers = y[inliersIdx]
    
    # Split inliers into train and test sets
    XTrain, XTestInliers, yTrain, yTestInliers = train_test_split(XInliers, yInliers, test_size=0.1, random_state=42)
    
    # Add outliers to the test set
    XTest = np.vstack([XTestInliers, XOutliers])
    yTest = np.hstack([yTestInliers, yOutliers])
    
    # Initialize and fit the ADDaMM model
    model = SimplifiedADDaMM(bandwidth=0.06)
    model.fit(XTrain)
    logger.info("Model fitted, starting testing...")
    
    # Make predictions on the test set
    logProbTest = model.detect(np.array(XTest))
    yPredTest = np.where(logProbTest < 0, 1, 0)
    
    # Make predictions on the training set
    logProbTrain = model.detect(XTrain)
    yPredTrain = np.where(logProbTrain < 0, 1, 0)
    
    # Save confusion matrices as images
    saveConfusionMatrix(yTest, yPredTest, 'confusion_matrix_test.png')
    saveConfusionMatrix(yTrain, yPredTrain, 'confusion_matrix_train.png')
    
    # Plot log likelihood scores
    plotScores(XTest, yTest, logProbTest, 'log_likelihood_scores.png')
    
    # Log performance metrics for the test set
    accuracy = accuracy_score(yTest, yPredTest)
    precision = precision_score(yTest, yPredTest)
    recall = recall_score(yTest, yPredTest)
    f1 = f1_score(yTest, yPredTest)
    
    logger.info(f"Test Data - Accuracy: {accuracy}")
    logger.info(f"Test Data - Precision: {precision}")
    logger.info(f"Test Data - Recall: {recall}")
    logger.info(f"Test Data - F1 Score: {f1}")
    logger.info("Test Data - Classification Report:")
    logger.info(classification_report(yTest, yPredTest))
    
    # Log performance metrics for the training set
    accuracyTrain = accuracy_score(yTrain, yPredTrain)
    precisionTrain = precision_score(yTrain, yPredTrain)
    recallTrain = recall_score(yTrain, yPredTrain)
    f1Train = f1_score(yTrain, yPredTrain)
    
    logger.info(f"Train Data - Accuracy: {accuracyTrain}")
    logger.info(f"Train Data - Precision: {precisionTrain}")
    logger.info(f"Train Data - Recall: {recallTrain}")
    logger.info(f"Train Data - F1 Score: {f1Train}")
    logger.info("Train Data - Classification Report:")
    logger.info(classification_report(yTrain, yPredTrain))

# Main function
def main(filePath):
    logger.info("Loading data from CSV...")
    X, y = loadDataFromCsv(filePath)
    
    logger.info("Preprocessing data...")
    XScaled = preprocessData(X)
    
    logger.info("Evaluating model...")
    evaluateModel(XScaled, y)
    
    logger.info("Script completed successfully.")

# Run the script
if __name__ == "__main__":
    filePath = 'datasets/KDD2014_donors_10feat_nomissing_normalised.csv'  # Replace with your CSV file path
    main(filePath)
