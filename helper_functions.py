"""
Helper functions for the text classification project.
"""
import string
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_score, \
    recall_score, f1_score
import re
from typing import Tuple, List, Union, Any, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


def inspect_dataframe(my_df: pd.DataFrame):
    """
    Inspect the DataFrame and print out detailed information about its structure and size.

    Args:
    - df (pd.DataFrame): The DataFrame to inspect.

    Returns:
    - None
    """
    df = my_df.copy(deep=True)

    try:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    except ValueError as e:
        print(f"Error converting 'date' column to datetime: {e}")

    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nColumn names and data types:")
    print(df.dtypes)

    missing_values_count = df.isnull().sum()
    missing_values_percentage = (missing_values_count / len(df)) * 100
    print("\nMissing values for each column:")
    for col in missing_values_count.index:
        print(f"{col}: {missing_values_count[col]} missing values ({missing_values_percentage[col]:.2f}%)")

    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        print("\nCategorical columns:")
        for col in categorical_columns:
            print(f"\nColumn: {col}")
            print(f"Count of unique values: {df[col].nunique()}")
            if col not in ['title', 'text']:
                print(f"Mode: {df[col].mode()[0]}")
                print(f"Unique values (up to the first 10): {df[col].unique().tolist()[:10]}")

            if col in ['title', 'text']:
                duplicate_count = df.duplicated(subset=[col]).sum()
                duplicate_percentage = (duplicate_count / len(df)) * 100
                print(f"Count of duplicate values: {duplicate_count} ({duplicate_percentage:.2f}%)")

    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_columns) > 0:
        print("\nDatetime columns:")
        for col in datetime_columns:
            print(f"\nColumn: {col}")
            print(f"  Minimum date: {df[col].min()}")
            print(f"  Maximum date: {df[col].max()}")
            print(f"  Mode: {df[col].mode()[0]}")


def print_duplicate_statistics(my_df: pd.DataFrame) -> None:
    """
    Print statistics related to duplicate rows and columns in a DataFrame.

    Args:
    - my_df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """
    df = my_df.copy(deep=True)

    total_rows = len(df)
    all_duplicates_count = df.duplicated().sum()
    all_duplicates_percentage = (all_duplicates_count / total_rows) * 100

    title_duplicates_count = df['title'].duplicated().sum()
    title_duplicates_percentage = (title_duplicates_count / total_rows) * 100

    text_duplicates_count = df['text'].duplicated().sum()
    text_duplicates_percentage = (text_duplicates_count / total_rows) * 100

    combined_title_text = df['title'] + df['text']
    combined_duplicates_count = combined_title_text.duplicated().sum()
    combined_duplicates_percentage = (combined_duplicates_count / total_rows) * 100

    combined_df = df.copy()
    combined_df['combined_title_text'] = combined_title_text

    title_duplicates_not_in_combined = combined_df['title'].duplicated() & ~combined_df[
        'combined_title_text'].duplicated()
    text_duplicates_not_in_combined = combined_df['text'].duplicated() & ~combined_df[
        'combined_title_text'].duplicated()

    title_duplicates_not_in_combined_count = title_duplicates_not_in_combined.sum()
    title_duplicates_not_in_combined_percentage = (title_duplicates_not_in_combined_count / total_rows) * 100

    text_duplicates_not_in_combined_count = text_duplicates_not_in_combined.sum()
    text_duplicates_not_in_combined_percentage = (text_duplicates_not_in_combined_count / total_rows) * 100

    print("Total number of rows: ", total_rows)
    print("-" * 20)
    print("Number of completely duplicate rows in the dataset:", all_duplicates_count)
    print("Percentage of completely duplicate rows in the dataset: {:.2f}%".format(all_duplicates_percentage))
    print("-" * 20)
    print("Number of duplicate values for 'title' column:", title_duplicates_count)
    print("Percentage of duplicate values for 'title' column: {:.2f}%".format(title_duplicates_percentage))
    print("-" * 20)
    print("Number of duplicate values for 'text' column:", text_duplicates_count)
    print("Percentage of duplicate values for 'text' column: {:.2f}%".format(text_duplicates_percentage))
    print("-" * 20)
    print("Number of duplicate values for combined 'title' and 'text' columns:", combined_duplicates_count)
    print("Percentage of duplicate values for combined 'title' and 'text' columns: {:.2f}%".format(
        combined_duplicates_percentage))
    print("-" * 20)
    print("Number of duplicate values for 'title' column not in combined duplicates:",
          title_duplicates_not_in_combined_count)
    print("Percentage of duplicate values for 'title' column not in combined duplicates: {:.2f}%".format(
        title_duplicates_not_in_combined_percentage))
    print("-" * 20)
    print("Number of duplicate values for 'text' column not in combined duplicates:",
          text_duplicates_not_in_combined_count)
    print("Percentage of duplicate values for 'text' column not in combined duplicates: {:.2f}%".format(
        text_duplicates_not_in_combined_percentage))
    print("-" * 50)

    duplicate_rows = combined_df[combined_df.duplicated()]
    double_duplicates = combined_df[combined_df['combined_title_text'].duplicated()]

    print("\nDistribution of 'subject' among duplicate values:")
    print(duplicate_rows['subject'].value_counts())
    print("-" * 20)
    print("\nDistribution of 'date' among duplicate values:")
    print(duplicate_rows['date'].value_counts())
    print("-" * 20)
    print("\nDistribution of 'label' among duplicate values:")
    print(duplicate_rows['label'].value_counts())
    print("-" * 20)
    print("Distribution of 'subject' among double duplicates:")
    print(double_duplicates['subject'].value_counts())
    print("-" * 20)
    print("\nDistribution of 'date' among double duplicates:")
    print(double_duplicates['date'].value_counts())
    print("-" * 20)
    print("\nDistribution of 'label' among double duplicates:")
    print(double_duplicates['label'].value_counts())


def clean_text(text):
    """
    Clean the text in a string.
    The function makes the string lowercase,
    removes white spaces, non-ASCII characters, links,
    date references, source references, and emojis.
    Also, it removes words leading up to '(Reuters)'.
    """
    text = re.sub(r'.*\(Reuters\)', '', text)
    text = re.sub(r' \(Reuters\) - ', ' ', text)
    text = re.sub(r'^ - ', '', text)
    text = text.replace('(Reuters)', '')
    text = text.replace('reuters', '')
    text = text.replace('politifact', '')
    text = text.replace('21st century wire', '')
    text = text.lower()
    text = ' '.join(text.split())
    text = ''.join(char for char in text if ord(char) < 128)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b(20\d{2}|19\d{2})\b', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    return text


def extract_features(text: str) -> Tuple[int, int, int]:
    """
    Function to extract features from the input text.
    """
    num_sentences = text.count('.')
    num_words = len(text.split())
    unique_words = len(set(text.split()))
    return num_sentences, num_words, unique_words


def plot_violin_subplots(x: str,
                         y_list: List[str],
                         df: pd.DataFrame,
                         titles: Union[List[str], None] = None,
                         figsize: Union[tuple, None] = (15, 10)):
    """
    Plot violin subplots for multiple variables.

    Args:
    - x (str): The column name for the x-axis.
    - y_list (List[str]): A list of column names for the y-axis.
    - df (pd.DataFrame): The DataFrame containing the data.
    - titles (List[str], optional): A list of titles for each subplot. If not provided, default titles will be used. Defaults to None.
    - figsize (tuple, optional): The size of the figure. Defaults to (15, 10).

    Returns:
    - None: Displays the violin subplots.
    """
    num_subplots = len(y_list)
    fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
    for i in range(num_subplots):
        title = f"Distribution of {titles[i]}" if titles else f'Distribution of {y_list[i]} by {x}'
        sns.violinplot(x=x, y=y_list[i], data=df, ax=axes[i], palette=['blue', 'red'])
        axes[i].set_title(title)
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def preprocess_text(text: str, remove_stopwords: bool = True, remove_punctuation: bool = True) -> str:
    """
    Preprocess the text by removing stopwords, punctuation, and lemmatizing.

    Args:
    - text (str): Input text to be preprocessed.
    - remove_stopwords (bool): Whether to remove stopwords or not.
    - remove_punctuation (bool): Whether to remove punctuation or not.

    Returns:
    - str: Preprocessed text.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    if remove_stopwords:
        tokens = [lemmatizer.lemmatize(token) for token in tokens
                  if token.lower() not in stopwords.words('english')]
    else:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    if remove_punctuation:
        tokens = [token.translate(str.maketrans('', '',
                                                string.punctuation)) for token in tokens]
    return ' '.join(tokens[:1024])


def preprocess_ml(text_series: pd.Series,
                  remove_stopwords: bool = True,
                  remove_punctuation: bool = True):
    """
    Preprocess data for traditional ML models.
    """
    return text_series.apply(lambda text: preprocess_text(text, remove_stopwords=remove_stopwords,
                                                          remove_punctuation=remove_punctuation))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot the confusion matrix.

    Args:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.

    Returns:
    - None
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_auc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Plot the ROC AUC curve.

    Args:
    - y_true (np.ndarray): True labels.
    - y_proba (np.ndarray): Predicted probabilities.

    Returns:
    - None
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.show()


def train_and_evaluate(classifier: Any,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       fit: bool = True,
                       X_train: np.ndarray = None,
                       y_train: np.ndarray = None,
                       plot_conf_matrix: bool = False,
                       plot_roc_curve: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """
    Train and evaluate the classifier on the validation set.

    Args:
    - classifier (Any): The classifier model to be trained and evaluated.
    - X_train (np.ndarray): Training features.
    - X_val (np.ndarray): Validation features.
    - y_train (np.ndarray): Training labels.
    - y_val (np.ndarray): Validation labels.
    - fit (bool): Whether to fit the classifier. Default is True.
    - plot_confusion_matrix (bool): Whether to plot the confusion matrix. Default is False.
    - plot_roc_curve (bool): Whether to plot the ROC AUC curve. Default is False.

    Returns:
    - y_pred (np.ndarray): Predictions on the validation set.
    - y_proba (np.ndarray): Predicted probabilities on the validation set.
    - Trained classifier if fit=True, otherwise None.
    """
    if fit:
        classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)
    print(classification_report(y_val, y_pred))

    y_proba = classifier.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    print(f'ROC AUC Score: {roc_auc}')

    if plot_conf_matrix:
        plot_confusion_matrix(y_val, y_pred)

    if plot_roc_curve:
        plot_roc_auc_curve(y_val, y_proba)

    return y_pred, y_proba, (classifier if fit else None)


def objective(trial: optuna.Trial, 
              X_train: csr_matrix, 
              y_train: Union[List[int], csr_matrix], 
              X_val: csr_matrix, 
              y_val: Union[List[int], csr_matrix], 
              classifier: ClassifierMixin, 
              feature_ranges: Dict[str, Union[int, float, List[int], List[float]]]) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (csr_matrix): Training feature matrix.
        y_train (Union[List[int], csr_matrix]): Training labels.
        X_val (csr_matrix): Validation feature matrix.
        y_val (Union[List[int], csr_matrix]): Validation labels.
        classifier (ClassifierMixin): Classifier object.
        feature_ranges (Dict[str, Union[int, float, List[int], List[float]]]): Dictionary defining the search space
            for hyperparameters.

    Returns:
        float: ROC-AUC score of the classifier on the validation data.
    """
    params = {}
    for param, (low, high) in feature_ranges.items():
        if isinstance(low, int) and isinstance(high, int):
            params[param] = trial.suggest_int(param, low, high)
        elif isinstance(low, float) and isinstance(high, float):
            params[param] = trial.suggest_float(param, low, high)
        elif isinstance(low, float) and isinstance(high, int):
            params[param] = trial.suggest_discrete_uniform(param, low, high, 1)
        elif isinstance(low, list) and isinstance(high, list):
            params[param] = trial.suggest_categorical(param, [i for i in range(len(low), len(high))])
        else:
            raise ValueError(f"Unsupported parameter type for {param}")

    clf = classifier.set_params(**params)
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc


def optimize_classifier(classifier_name: str, 
                        classifier: ClassifierMixin, 
                        X_train: csr_matrix, 
                        y_train: Union[List[int], csr_matrix], 
                        X_val: csr_matrix, 
                        y_val: Union[List[int], csr_matrix], 
                        feature_ranges: Dict[str, Union[int, float, List[int], List[float]]], 
                        trials: int = 100) -> Tuple[ClassifierMixin, float]:
    """
    Optimize hyperparameters of a classifier using Optuna.

    Args:
        classifier_name (str): Name of the classifier.
        classifier (ClassifierMixin): Classifier object to optimize.
        X_train (csr_matrix): Training feature matrix.
        y_train (Union[List[int], csr_matrix]): Training labels.
        X_val (csr_matrix): Validation feature matrix.
        y_val (Union[List[int], csr_matrix]): Validation labels.
        feature_ranges (Dict[str, Union[int, float, List[int], List[float]]]): Dictionary defining the search space
            for hyperparameters.

    Returns:
        Tuple[ClassifierMixin, float]: Best optimized classifier and its ROC-AUC score on the validation data.
    """
    print(f"Optimizing {classifier_name}:")
    study = optuna.create_study(direction='maximize')
    objective_fn = lambda trial: objective(trial, X_train, y_train, X_val, y_val, classifier, feature_ranges)
    study.optimize(objective_fn, n_trials=trials)
    best_params = study.best_params
    best_classifier = classifier.set_params(**best_params)
    
    best_classifier.fit(X_train, y_train)
    y_pred_proba = best_classifier.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"Best hyperparameters for {classifier_name}:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"Best {classifier_name} ROC-AUC: {roc_auc:.4f}")
    return best_classifier, roc_auc


def get_model_last_layers(encoder_layer_names, model_name, num_last_layers=2):
    """
    Get the last layers of the pre-trained model.
    """
    if encoder_layer_names:
        encoder_layer_indices = [int(name.split('.')[1]) 
                                 for name in encoder_layer_names]
        unique_encoder_layer_indices = sorted(set(encoder_layer_indices), 
                                              reverse=True)
        
        last_two_encoder_layer_indices = unique_encoder_layer_indices[:num_last_layers]
        
        last_two_encoder_layer_names = [f"layer.{index}" 
                                        for index in last_two_encoder_layer_indices]
        
        print(f"Last two trainable base layers in the {model_name} model:",
              last_two_encoder_layer_names)
        return last_two_encoder_layer_names
    else:
        print("No encoder layers found in the model.")


def test_model(model, test_dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model = model.to(device)
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, true_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            true_labels = true_labels.to(device)

            outputs = model(input_ids, attention_mask)

            if model.model_name == 'bigbird':
                true_labels = true_labels.unsqueeze(1)

            if model.additional_fc:
                logits = outputs[:, 0]
            else:
                logits = outputs

            preds = torch.sigmoid(logits)

            all_targets.append(true_labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    return all_targets, all_predictions


def calculate_metrics(y_true, y_prob, thresholds=np.arange(0.1, 1.0, 0.01)):
    precisions, recalls, f1_scores, roc_auc_scores = [], [], [], []
    metrics = []
    y_prob = y_prob.mean(axis=1)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division='warn')
        recall = recall_score(y_true, y_pred, zero_division='warn')
        f1 = f1_score(y_true, y_pred, zero_division='warn')
        roc_auc = roc_auc_score(y_true, y_prob)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)
        metrics.append((threshold, precision, recall, f1, roc_auc))

    combined_scores = [f1 * roc_auc for f1, roc_auc in zip(f1_scores,
                                                           roc_auc_scores)]

    best_idx = np.argmax(combined_scores)
    best_threshold, best_precision, best_recall, best_f1, best_roc_auc = metrics[best_idx]

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f}," +
          f"F1-score: {best_f1:.4f}, ROC-AUC: {best_roc_auc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1-score')
    plt.plot(thresholds, roc_auc_scores, label='ROC-AUC')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1-score, ROC-AUC vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    return metrics