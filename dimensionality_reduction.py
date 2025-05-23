import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def perform_pca_from_csv(input_csv_path, output_csv_path, n_components=None):
    """
    Performs PCA on data from a CSV file and saves the results to another CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file.
        n_components (int, optional): Number of principal components to retain. 
                                     If None, all components are retained. Defaults to None.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'")
        return
    
    # Separate features for PCA
    X = df.select_dtypes(include=['number'])

    # Standardize the features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_scaled = X
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create a new DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components,
                          columns=[f'principal_component_{i+1}' for i in range(principal_components.shape[1])])

    # Combine with non-numeric columns, if any
    non_numeric_df = df.select_dtypes(exclude=['number'])
    if not non_numeric_df.empty:
        pca_df = pd.concat([non_numeric_df.reset_index(drop=True), pca_df], axis=1)

    # Save the result to a new CSV file
    pca_df.to_csv(output_csv_path, index=False)
    print(f"PCA results saved to '{output_csv_path}'")

    return pca_df
