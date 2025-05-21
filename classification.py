import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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