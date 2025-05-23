from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import utils

def run_knn_on_csv(
    input_csv_path,
    target_column,
    output_csv_path,
    n_neighbors= 5,
    test_size=0.3,
    random_state= 42
):
    """
    Performs K-Nearest Neighbors (KNN) classification on data from a CSV file,
    saves the results to a new CSV, and returns the predicted labels.

    Args:
        input_csv_path (str): The path to the input CSV file.
        target_column (str): The name of the column to be used as the target variable (y).
        n_neighbors (int): The number of neighbors to use for KNN. Defaults to 5.
        test_size (float): The proportion of the dataset to include in the test split.
                           Defaults to 0.3 (30%).
        random_state (int): Controls the shuffling applied to the data before applying
                            the split. Pass an int for reproducible output across multiple
                            function calls. Defaults to 42.
        output_csv_path (str): The path where the predictions CSV will be saved.
                               Defaults to "knn_predictions.csv".

    Returns:
        pd.Series: A pandas Series containing the predicted labels for the test set.
    """
    try:
        # 1. Load the dataset from CSV
        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")
        print(f"Original data shape: {df.shape}")

        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the CSV file.")

        # 2. Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")


        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        ) # stratify=y ensures that the proportion of target variable is maintained in train and test sets

        print(f"Training data shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

        # 4. Initialize and train the KNN classifier
        print(f"Initializing KNN classifier with n_neighbors={n_neighbors}...")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        print("KNN model trained successfully.")

        # 5. Make predictions on the test set
        print("Making predictions on the test set...")
        y_pred = knn.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predicted_" + target_column)
        print("Predictions made.")
        print(y_pred_series)

        # 6. Evaluate the model (optional, but good for understanding performance)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))


        # 7. Prepare the output DataFrame
        # Combine original test features with actual and predicted labels
        output_df = X_test.copy()
        output_df[f"Actual_{target_column}"] = y_test
        output_df[f"Predicted_{target_column}"] = y_pred

        # 8. Save the results to a new CSV file
        print(f"\nSaving results to: {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)
        print("Results saved successfully.")

        
        # 5. Generate the Confusion Matrix
        # This compares the true labels (y_test) with the predicted labels (y_pred).
        # Each row represents instances in an actual class, while each column represents instances in a predicted class.
        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix (Raw Data):")
        print(cm)

        # 6. Display the Confusion Matrix using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,      # Show the actual numbers in each cell
                    fmt='d',         # Format numbers as integers
                    cmap='Blues',    # Color map for the heatmap (e.g., 'Blues', 'Greens', 'YlGnBu')
                    xticklabels=output_df[f"Predicted_{target_column}"], # Labels for predicted classes on x-axis
                    yticklabels=output_df[f"Actual_{target_column}"]) # Labels for true classes on y-axis
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for KNN Classifier (n_neighbors={n_neighbors})')
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()

        # Alternatively, using ConfusionMatrixDisplay (from scikit-learn, often preferred)
        # This provides a more convenient and often better-looking way to plot directly.
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_column)
        disp.plot(cmap='Blues', values_format='d') # 'd' for integer format
        plt.title(f'Confusion Matrix for KNN Classifier (n_neighbors={n_neighbors})')
        plt.tight_layout()
        plt.show()

        print("\nConfusion matrices displayed successfully.")

        return output_df

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return pd.Series([])
    except ValueError as e:
        print(f"Error: {e}")
        return pd.Series([])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.Series([])
    
def run_naive_bayes_on_csv(
    input_csv_path: str,
    target_column: str,
    output_csv_path:str,
    test_size: float = 0.3,
    random_state: int = 42
) -> pd.Series:
    """
    Performs Naive Bayes classification on data from a CSV file,
    saves the results to a new CSV, and returns the predicted labels.

    Args:
        input_csv_path (str): The path to the input CSV file.
        target_column (str): The name of the column to be used as the target variable (y).
        test_size (float): The proportion of the dataset to include in the test split.
                           Defaults to 0.3 (30%).
        random_state (int): Controls the shuffling applied to the data before applying
                            the split. Pass an int for reproducible output across multiple
                            function calls. Defaults to 42.
        output_csv_path (str): The path where the predictions CSV will be saved.
                               Defaults to "naive_bayes_predictions.csv".

    Returns:
        pd.Series: A pandas Series containing the predicted labels for the test set.
    """
    try:
        # 1. Load the dataset from CSV
        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")
        print(f"Original data shape: {df.shape}")

        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the CSV file.")

        # 2. Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")

        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        ) # stratify=y ensures that the proportion of target variable is maintained in train and test sets

        print(f"Training data shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

        # 4. Initialize and train the Naive Bayes classifier
        print(f"Initializing Naive Bayes classifier (GaussianNB)...")
        gnb = GaussianNB() # Using Gaussian Naive Bayes for continuous features
        gnb.fit(X_train, y_train)
        print("Naive Bayes model trained successfully.")

        # 5. Make predictions on the test set
        print("Making predictions on the test set...")
        y_pred = gnb.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predicted_" + target_column)
        print("Predictions made.")
        print(y_pred_series)

        # 6. Evaluate the model (optional, but good for understanding performance)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))


        # 7. Prepare the output DataFrame
        # Combine original test features with actual and predicted labels
        output_df = X_test.copy()
        output_df[f"Actual_{target_column}"] = y_test
        output_df[f"Predicted_{target_column}"] = y_pred

        # 8. Save the results to a new CSV file
        print(f"\nSaving results to: {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)
        print("Results saved successfully.")

        return output_df

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return pd.Series([])
    except ValueError as e:
        print(f"Error: {e}")
        return pd.Series([])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.Series([])
    
def run_svm_on_csv(
    input_csv_path: str,
    target_column: str,
    output_csv_path: str, # Updated default output path
    test_size: float = 0.3,
    random_state: int = 42,
    C: float = 1.0, # SVM regularization parameter
    kernel: str = 'rbf' # Specifies the kernel type to be used in the algorithm
) -> pd.Series:
    """
    Performs Support Vector Machine (SVM) classification on data from a CSV file,
    saves the results to a new CSV, and returns the predicted labels.

    Args:
        input_csv_path (str): The path to the input CSV file.
        target_column (str): The name of the column to be used as the target variable (y).
        test_size (float): The proportion of the dataset to include in the test split.
                           Defaults to 0.3 (30%).
        random_state (int): Controls the shuffling applied to the data before applying
                            the split. Pass an int for reproducible output across multiple
                            function calls. Defaults to 42.
        output_csv_path (str): The path where the predictions CSV will be saved.
                               Defaults to "svm_predictions.csv".
        C (float): Regularization parameter. The strength of the regularization is
                   inversely proportional to C. Must be strictly positive.
                   Defaults to 1.0.
        kernel (str): Specifies the kernel type to be used in the algorithm.
                      It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
                      or a callable. Defaults to 'rbf'.

    Returns:
        pd.Series: A pandas Series containing the predicted labels for the test set.
    """
    try:
        # 1. Load the dataset from CSV
        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")
        print(f"Original data shape: {df.shape}")

        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the CSV file.")

        # 2. Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")

        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        ) # stratify=y ensures that the proportion of target variable is maintained in train and test sets

        print(f"Training data shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

        # 4. Initialize and train the SVM classifier
        print(f"Initializing SVM classifier with C={C}, kernel='{kernel}'...")
        svm_classifier = SVC(C=C, kernel=kernel, random_state=random_state) # Using SVC for SVM
        svm_classifier.fit(X_train, y_train)
        print("SVM model trained successfully.")

        # 5. Make predictions on the test set
        print("Making predictions on the test set...")
        y_pred = svm_classifier.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predicted_" + target_column)
        print("Predictions made.")
        print(y_pred_series)

        # 6. Evaluate the model (optional, but good for understanding performance)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        # Generate classification report as a dictionary
        print(classification_report(y_test, y_pred))

        # 7. Prepare the output DataFrame
        # Combine original test features with actual and predicted labels
        output_df = X_test.copy()
        output_df[f"Actual_{target_column}"] = y_test
        output_df[f"Predicted_{target_column}"] = y_pred

        # 8. Save the results to a new CSV file
        print(f"\nSaving results to: {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)
        print("Results saved successfully.")

        return output_df

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return pd.Series([])
    except ValueError as e:
        print(f"Error: {e}")
        return pd.Series([])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.Series([])
    
def run_id3_on_csv(
    input_csv_path: str,
    target_column: str,
    output_csv_path: str, # Updated default output path
    test_size: float = 0.3,
    random_state: int = 42,
    criterion: str = 'entropy', # Criterion for information gain (ID3 typically uses entropy)
    max_depth: int = None # Maximum depth of the tree (None means unlimited)
) -> pd.Series:
    """
    Performs ID3 (Decision Tree) classification on data from a CSV file,
    saves the results to a new CSV, and returns the predicted labels.

    Args:
        input_csv_path (str): The path to the input CSV file.
        target_column (str): The name of the column to be used as the target variable (y).
        test_size (float): The proportion of the dataset to include in the test split.
                           Defaults to 0.3 (30%).
        random_state (int): Controls the shuffling applied to the data before applying
                            the split. Pass an int for reproducible output across multiple
                            function calls. Defaults to 42.
        output_csv_path (str): The path where the predictions CSV will be saved.
                               Defaults to "id3_predictions.csv".
        criterion (str): The function to measure the quality of a split.
                         "entropy" for information gain (typical for ID3).
                         "gini" for the Gini impurity. Defaults to 'entropy'.
        max_depth (int): The maximum depth of the tree. If None, then nodes are expanded
                         until all leaves are pure or until all leaves contain less than
                         min_samples_split samples. Defaults to None.

    Returns:
        pd.Series: A pandas Series containing the predicted labels for the test set.
    """
    try:
        # 1. Load the dataset from CSV
        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")
        print(f"Original data shape: {df.shape}")

        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the CSV file.")

        # 2. Separate features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")

        # 3. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        ) # stratify=y ensures that the proportion of target variable is maintained in train and test sets

        print(f"Training data shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

        # 4. Initialize and train the ID3 (Decision Tree) classifier
        print(f"Initializing ID3 (Decision Tree) classifier with criterion='{criterion}', max_depth={max_depth}...")
        id3_classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
        id3_classifier.fit(X_train, y_train)
        print("ID3 (Decision Tree) model trained successfully.")

        # 5. Make predictions on the test set
        print("Making predictions on the test set...")
        y_pred = id3_classifier.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predicted_" + target_column)
        print("Predictions made.")
        print(y_pred_series)

        # 6. Evaluate the model (optional, but good for understanding performance)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))



        # 7. Prepare the output DataFrame
        # Combine original test features with actual and predicted labels
        output_df = X_test.copy()
        output_df[f"Actual_{target_column}"] = y_test
        output_df[f"Predicted_{target_column}"] = y_pred

        # 8. Save the results to a new CSV file
        print(f"\nSaving results to: {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)
        print("Results saved successfully.")

        return output_df

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return pd.Series([])
    except ValueError as e:
        print(f"Error: {e}")
        return pd.Series([])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.Series([])