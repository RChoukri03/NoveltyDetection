
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from addamm import SimplifiedADDaMM  # Assurez-vous que ce module est accessible
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sélection des métriques spécifiques
def iac_metrics_only(df):
    return df[[
        'failure_prone',
        'avg_play_size', 'avg_task_size', 'lines_blank', 'lines_code', 'lines_comment', 'num_authorized_key', 'num_block_error_handling', 'num_blocks', 'num_commands',
        'num_conditions', 'num_decisions', 'num_deprecated_keywords', 'num_deprecated_modules', 'num_distinct_modules', 'num_external_modules', 'num_fact_modules',
        'num_file_exists', 'num_file_mode', 'num_file_modules', 'num_filters', 'num_ignore_errors', 'num_import_playbook', 'num_import_role', 'num_import_tasks',
        'num_include', 'num_include_role', 'num_include_tasks', 'num_include_vars', 'num_keys', 'num_lookups', 'num_loops', 'num_math_operations', 'num_names_with_vars',
        'num_parameters', 'num_paths', 'num_plays', 'num_regex', 'num_roles', 'num_suspicious_comments', 'num_tasks', 'num_tokens', 'num_unique_names', 'num_uri', 'num_prompts',
        'num_vars', 'text_entropy']]


def preprocessData(X):
    logger.info("Starting data preprocessing...")
    # quantileScaler = QuantileTransformer(n_quantiles=335,output_distribution='normal')
    standardScaler = StandardScaler()

    # Désactiver le prétraitement pour le moment
    # X = quantileScaler.fit_transform(X)
    X = standardScaler.fit_transform(X)
    logger.info("Data preprocessing completed.")
    
    return X

# Charger les données CSV et appliquer la sélection de métriques
def loadAndProcessData(release_type, metric_function):
    filePath = f'datasets/ansible/ansible_{release_type}_releases.csv'
    df = pd.read_csv(filePath).fillna(0).select_dtypes(include='number')
    
    logger.info(f"Loaded dataset for {release_type} release with {len(df)} samples.")
    df = metric_function(df)  # Appliquer la fonction de sélection de métriques
    
    Xpos = df[df.failure_prone == 0].drop(['failure_prone'], axis=1)  # Classe normale (clean)
    Xneg = df[df.failure_prone == 1].drop(['failure_prone'], axis=1)  # Classe anomale (failure-prone)
    Xpos = preprocessData(Xpos)
    Xneg = preprocessData(Xneg)
    return Xpos, Xneg

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

# Évaluation du modèle
def evaluateModel(Xpos, Xneg, release_type):
    logger.info("Starting model evaluation...")
    
    X_train = Xpos  # Entraîner uniquement sur les inliers (classe normale)
    X_test = np.vstack([Xpos, Xneg])  # Test sur inliers et outliers
    y_test = np.hstack([np.zeros(len(Xpos)), np.ones(len(Xneg))])  # 0 = inliers, 1 = outliers
    
    # Initialize and fit the ADDaMM model
    model = SimplifiedADDaMM(bandwidth=0.01)
    model.fit(X_train)
    logger.info("Model fitted, start Testing ...")
    
    # Make predictions on the test set
    log_prob_test = model.detect(np.array(X_test))
    y_pred_test = np.where(log_prob_test < 0, 1, 0)  # 1 = outliers, 0 = inliers
    
    # Save confusion matrix as image
    saveConfusionMatrix(y_test, y_pred_test, f'datasets/ansible/confusion_matrix_test_{release_type}.png')
    
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

# Fonction principale
def main():
    for release_type in ['last', 'midst']:
        logger.info(f"Processing {release_type} release...")
        
        # Sélectionner la fonction de métrique à utiliser (ex: iac_metrics_only)
        Xpos, Xneg = loadAndProcessData(release_type, iac_metrics_only)  # Vous pouvez remplacer par delta_metrics_only ou process_metrics_only
        
        logger.info("Evaluating model...")
        evaluateModel(Xpos, Xneg, release_type)
    
    logger.info("Script completed successfully.")

# Exécution du script
if __name__ == "__main__":
    main()
