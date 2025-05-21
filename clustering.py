import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


def perform_kmeans_from_csv(input_csv_path, output_csv_path, n_clusters):
    # 1. Read CSV file
    df = pd.read_csv(input_csv_path)

    # 2. Identify Numeric Columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        print("Error: No numeric columns found in the CSV file.")
        return

    print(f"Using the following numeric columns for clustering: {numeric_cols}")
    X = df[numeric_cols].copy() # Create a copy to avoid modifying the original DataFrame

    # 3. Handle Missing Values (if any) - fill with the mean for numeric columns
    X = X.fillna(X.mean())

    # 4. Feature Scaling (optional but often recommended)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 6. Save to CSV
    df.to_csv(output_csv_path, index=False)

    print(f"Clustering complete. Results saved to '{output_csv_path}'")

    return df

def perform_agglomerative_from_csv(input_csv_path, output_csv_path, n_clusters=2):
        # 1. Load Data
        df = pd.read_csv(input_csv_path)

        # 2. Identify Numeric Columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            print("Error: No numeric columns found in the CSV file.")
            return

        print(f"Using the following numeric columns for clustering: {numeric_cols}")
        X = df[numeric_cols].copy()

        # 3. Handle Missing Values (fill with the mean for numeric columns)
        X = X.fillna(X.mean())

        # 4. Feature Scaling (optional but often recommended)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5. Apply Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_clustering.fit_predict(X_scaled)

        # 6. Add Cluster Labels to DataFrame
        df['cluster'] = cluster_labels

        # 7. Print the Results
        print("\nAgglomerative Clustering Results:")
        print(df[['cluster'] + numeric_cols]) # Display cluster and the features used

        # 8. Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to '{output_csv_path}'")

        return df
def perform_dbscan_clustering_from_csv(input_csv_file,output_csv_file, eps=0.5, min_samples=5 ):
    """
    Applies DBSCAN clustering to a CSV file using all numeric columns,
    prints the cluster assignments, saves the results to a new CSV file,
    and displays the results in a Tkinter window.

    Args:
        input_csv_file (str): Path to the input CSV file.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.5.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Defaults to 5.
        output_csv_file (str, optional): Path to save the output CSV file.
                                         Defaults to 'dbscan_clusters.csv'.

    Returns:
        pandas.DataFrame: The DataFrame with the added 'cluster' column indicating DBSCAN cluster assignments.  Returns None on error.
    """
    try:
        # 1. Load Data
        df = pd.read_csv(input_csv_file)

        # 2. Identify Numeric Columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            print("Error: No numeric columns found in the CSV file.")
            return None

        print(f"Using the following numeric columns for clustering: {numeric_cols}")
        X = df[numeric_cols].copy()

        # 3. Handle Missing Values (fill with the mean for numeric columns)
        X = X.fillna(X.mean())

        # 4. Feature Scaling (optional but often recommended for DBSCAN)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        # 5. Apply DBSCAN Clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)

        # 6. Add Cluster Labels to DataFrame
        df['cluster'] = cluster_labels

        # 7. Print the Results
        print("\nDBSCAN Clustering Results:")
        print(df[['cluster'] + numeric_cols])  # Display cluster and the features used

        # 8. Save to CSV
        df.to_csv(output_csv_file, index=False)
        print(f"\nResults saved to '{output_csv_file}'")

        return df  # Return the modified DataFrame

    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
