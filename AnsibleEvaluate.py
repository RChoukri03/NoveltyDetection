import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from addamm import SimplifiedADDaMM  # Make sure this module is accessible
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to select specific metrics from the dataset
def iacMetricsOnly(df):
    return df[[  
        'failure_prone',
        'avg_play_size', 'avg_task_size', 'lines_blank', 'lines_code', 'lines_comment', 'num_authorized_key', 'num_block_error_handling', 
        'num_blocks', 'num_commands', 'num_conditions', 'num_decisions', 'num_deprecated_keywords', 'num_deprecated_modules', 
        'num_distinct_modules', 'num_external_modules', 'num_fact_modules', 'num_file_exists', 'num_file_mode', 'num_file_modules', 
        'num_filters', 'num_ignore_errors', 'num_import_playbook', 'num_import_role', 'num_import_tasks', 'num_include', 'num_include_role', 
        'num_include_tasks', 'num_include_vars', 'num_keys', 'num_lookups', 'num_loops', 'num_math_operations', 'num_names_with_vars', 
        'num_parameters', 'num_paths', 'num_plays', 'num_regex', 'num_roles', 'num_suspicious_comments', 'num_tasks', 'num_tokens', 
        'num_unique_names', 'num_uri', 'num_prompts', 'num_vars', 'text_entropy']]

# Function to preprocess the dataset
def preprocessData(X):
    logger.info("Starting data preprocessing...")
    scaler = StandardScaler()  # Standard scaling
    X = scaler.fit_transform(X)
    logger.info("Data preprocessing completed.")
    return X

# Load and process the dataset
def loadAndProcessData(releaseType, metricFunction):
    filePath = f'datasets/ansible/ansible_{releaseType}_releases.csv'
    df = pd.read_csv(filePath).fillna(0).select_dtypes(include='number')
    
    logger.info(f"Loaded dataset for {releaseType} release with {len(df)} samples.")
    df = metricFunction(df)  # Apply the metric selection function
    
    Xpos = df[df.failure_prone == 0].drop(['failure_prone'], axis=1)  # Normal class (non-failure)
    Xneg = df[df.failure_prone == 1].drop(['failure_prone'], axis=1)  # Anomalous class (failure-prone)
    Xpos = preprocessData(Xpos)
    Xneg = preprocessData(Xneg)
    return Xpos, Xneg

# Function to save the confusion matrix as an image
def saveConfusionMatrix(yTrue, yPred, filePath,releaseType):
    cm = confusion_matrix(yTrue, yPred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix {releaseType}')
    plt.savefig(filePath)
    logger.info(f"Confusion matrix saved as {filePath}")

# Model evaluation function
def evaluateModel(Xpos, Xneg, releaseType):
    logger.info("Starting model evaluation...")
    
    XTrain = Xpos  # Train on normal data (inliers)
    XTest = np.vstack([Xpos, Xneg])  # Test on both normal and anomalous data
    yTest = np.hstack([np.zeros(len(Xpos)), np.ones(len(Xneg))])  # Labels: 0 = inliers, 1 = outliers
    
    # Initialize and fit the ADDaMM model
    model = SimplifiedADDaMM(bandwidth=0.01)
    model.fit(XTrain)
    logger.info("Model fitted, starting testing...")
    
    # Make predictions
    logProbTest = model.detect(np.array(XTest))
    yPredTest = np.where(logProbTest < 0, 1, 0)  # 1 = outliers, 0 = inliers
    
    # Save confusion matrix as image
    saveConfusionMatrix(yTest, yPredTest, f'datasets/ansible/confusion_matrix_test_{releaseType}.png',releaseType)
    
    # Calculate and log performance metrics
    accuracy = accuracy_score(yTest, yPredTest)
    precision = precision_score(yTest, yPredTest)
    recall = recall_score(yTest, yPredTest)
    f1 = f1_score(yTest, yPredTest)
    
    logger.info(f"Test Data - Accuracy: {accuracy}")
    logger.info(f"Test Data - Precision: {precision}")
    logger.info(f"Test Data - Recall: {recall}")
    logger.info(f"Test Data - F1 Score: {f1}")
    
    # Print classification report
    logger.info("Test Data - Classification Report:")
    logger.info(classification_report(yTest, yPredTest))

# Main function
def main():
    for releaseType in ['last', 'midst']:
        logger.info(f"Processing {releaseType} release...")
        
        # Load and preprocess the data
        Xpos, Xneg = loadAndProcessData(releaseType, iacMetricsOnly)
        
        logger.info("Evaluating model...")
        evaluateModel(Xpos, Xneg, releaseType)
    
    logger.info("Script completed successfully.")

# Run the script
if __name__ == "__main__":
    main()
