import tkinter as tk
import customtkinter as ctk
import utils
import dimensionality_reduction as dr
import classification as cls
import clustering as clust


class MLKit:
    
    def __init__(self):
        self.root_window = ctk.CTk()
        self.root_window.title("ML Kit")
        self.__render()

    def run_pca(self):
        def submit_handler():
            df = dr.perform_pca_from_csv(filename, "./csv/output.csv",int(dimensions_entry.get()))
            utils.showDF(self.dimensionality_reduction_frame, df)

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.clustering_frame, bg="purple", padx=20, pady=20)
        dialog.geometry("280x100")
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter the number of attributes").grid(row=0, column=0,columnspan=2)
        dimensions_entry = tk.Entry(dialog, width=10)
        dimensions_entry.grid(row=1, column=0)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=1, column=1)
        dialog.mainloop()

    def run_kmeans(self):
        def submit_handler():
            df = clust.perform_kmeans_from_csv(filename, './csv/output_k.csv', int(cluster_entry.get()))
            utils.showDF(self.clustering_frame, df)

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.clustering_frame, bg="purple", padx=20, pady=20)
        dialog.geometry("280x100")
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter the number of clusters").grid(row=0, column=0,columnspan=2)
        cluster_entry = tk.Entry(dialog, width=10)
        cluster_entry.grid(row=1, column=0)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=1, column=1)
        dialog.mainloop()

    def run_agglomerative(self):
        filename = utils.load_csv_file()
        df = clust.perform_agglomerative_from_csv(filename, './csv/output_agg.csv')
        utils.showDF(self.clustering_frame, df)
        
    def run_dbscan(self):
        def submit_handler():
            df = clust.perform_dbscan_clustering_from_csv(filename,'./csv/output_dbscan.csv',eps=float(epsilon_entry.get()), min_samples=int(min_pts_entry.get()))
            utils.showDF(self.clustering_frame, df)

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.clustering_frame, bg="purple", padx=20, pady=20)
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter the parameters for DBSCAN Algorithms").grid(row=0, column=0, columnspan=2)
        tk.Label(dialog, bg="purple",fg="white", text="Epsilon").grid(row=1, column=0)
        tk.Label(dialog, bg="purple",fg="white", text="minPts").grid(row=2, column=0)
        epsilon_entry = tk.Entry(dialog,width=10)
        epsilon_entry.grid(row=1, column=1)
        min_pts_entry = tk.Entry(dialog,width=10)
        min_pts_entry.grid(row=2, column=1)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=3,column=0)
        dialog.mainloop()
        
    def run_knn(self):
        def submit_handler():
            df = cls.run_knn_on_csv(filename, target_column=target_column_entry.get(), n_neighbors=int(neighbor_entry.get()), output_csv_path="./csv/output_knn.csv")
            utils.showDF(self.classification_frame,df, title="Test Set")

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.classification_frame, bg="purple", padx=20, pady=20)
        dialog.geometry("280x100")
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter neighbors (n): ").grid(row=0, column=0)
        neighbor_entry = tk.Entry(dialog)
        neighbor_entry.grid(row=0, column=1)
        tk.Label(dialog, bg="purple",fg="white", text="Enter target column: ").grid(row=1, column=0)
        target_column_entry = tk.Entry(dialog)
        target_column_entry.grid(row=1, column=1)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=2, column=0)
        dialog.mainloop() 

    def run_naive_bayes_classifier(self):
        def submit_handler():
            df = cls.run_naive_bayes_on_csv(filename, target_column=target_column_entry.get(), output_csv_path="./csv/output_nbc.csv")
            utils.showDF(self.classification_frame,df, title="Test Set")

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.classification_frame, bg="purple", padx=20, pady=20)
        dialog.geometry("280x100")
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter target column: ").grid(row=0, column=0)
        target_column_entry = ctk.CTkEntry(dialog)
        target_column_entry.grid(row=0, column=1)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=1, column=0)
        dialog.mainloop()   
    
    def run_svm(self):
        def submit_handler():
            df = cls.run_svm_on_csv(filename, target_column=target_column_entry.get(), output_csv_path="./csv/output_svm.csv")
            utils.showDF(self.classification_frame,df, title="Test Set")

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.classification_frame, bg="purple", padx=20, pady=20)
        dialog.geometry("280x100")
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter target column: ").grid(row=0, column=0)
        target_column_entry = tk.Entry(dialog)
        target_column_entry.grid(row=0, column=1)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=1, column=0)
        dialog.mainloop()   
    
    def run_id3(self):
        def submit_handler():
            df = cls.run_id3_on_csv(filename, target_column=target_column_entry.get(), output_csv_path="./csv/output_id3.csv")
            utils.showDF(self.classification_frame,df, title="Test Set")

        filename = utils.load_csv_file()
        dialog = tk.Toplevel(self.classification_frame, bg="purple", padx=20, pady=20)
        dialog.geometry("280x100")
        dialog.title("Enter Parameters")
        tk.Label(dialog, bg="purple",fg="white", text="Enter target column: ").grid(row=0, column=0)
        target_column_entry = tk.Entry(dialog)
        target_column_entry.grid(row=0, column=1)
        tk.Button(dialog, text="Submit",command=submit_handler,padx=5, pady=2).grid(row=1, column=0)
        dialog.mainloop()   
    
    def __render(self):
        self.clustering_frame = tk.Frame(self.root_window,bg='lightblue',padx=20, pady=20)
        self.classification_frame = tk.Frame(self.root_window, bg='tan1',padx=20, pady=20)
        self.dimensionality_reduction_frame = tk.Frame(self.root_window, bg='lightgreen',padx=20, pady=20)
        self.clustering_frame.pack(fill=tk.X)
        self.classification_frame.pack(fill=tk.X)
        self.dimensionality_reduction_frame.pack(fill=tk.X)
        self.__render_clustering_frame()
        self.__render_classification_frame()
        self.__render_dimensionality_reduction_frame()
        self.root_window.mainloop()

    def __render_clustering_frame(self):
        ctk.CTkLabel(self.clustering_frame, text='CLUSTERING ALGORITHMS', justify="center", font=("Arial",24,"bold" )).grid(row=0, column=0, columnspan=2)
        ctk.CTkLabel(self.clustering_frame,  text='K-means++').grid(row=1, column=0)
        ctk.CTkLabel(self.clustering_frame,  text='Agglomerative').grid(row=2, column=0)
        ctk.CTkLabel(self.clustering_frame,  text='DBSCAN').grid(row=3, column=0)
        ctk.CTkButton(self.clustering_frame,text="Load CSV", command=self.run_kmeans).grid(row=1, column=1,padx=10, pady=5)
        ctk.CTkButton(self.clustering_frame, text="Load CSV", command=self.run_agglomerative).grid(row=2, column=1,padx=10, pady=5)
        ctk.CTkButton(self.clustering_frame, text="Load CSV", command=self.run_dbscan).grid(row=3, column=1,padx=10, pady=5)

    def __render_classification_frame(self):
        ctk.CTkLabel(self.classification_frame , text='CLASSIFICATION ALGORITHMS', justify="center", font=("Arial", 24,"bold" )).grid(row=0, column=0, columnspan=2)
        ctk.CTkLabel(self.classification_frame , text='KNN').grid(row=1, column=0)
        ctk.CTkLabel(self.classification_frame , text='Naive Bayesian Classifier').grid(row=2, column=0)
        ctk.CTkLabel(self.classification_frame , text='SVM').grid(row=3, column=0)
        ctk.CTkLabel(self.classification_frame , text='ID3').grid(row=4, column=0)
        ctk.CTkButton(self.classification_frame, text="Load CSV", command=self.run_knn).grid(row=1, column=1,padx=10, pady=5)
        ctk.CTkButton(self.classification_frame, text="Load CSV", command=self.run_naive_bayes_classifier).grid(row=2, column=1,padx=10, pady=5)
        ctk.CTkButton(self.classification_frame, text="Load CSV", command=self.run_svm).grid(row=3, column=1,padx=10, pady=5)
        ctk.CTkButton(self.classification_frame, text="Load CSV", command=self.run_id3).grid(row=4, column=1,padx=10, pady=5)

    def __render_dimensionality_reduction_frame(self):
        ctk.CTkLabel(self.dimensionality_reduction_frame,   text='DIMENSIONALITY REDUCTION', justify="center", font=("Arial", 24,"bold" )).grid(row=0, column=0, columnspan=3)
        ctk.CTkLabel(self.dimensionality_reduction_frame,  text='Principal Component Analysis').grid(row=1, column=0)
        ctk.CTkButton(self.dimensionality_reduction_frame, text='Load CSV', command=self.run_pca).grid(row=1, column=1,padx=10, pady=5)
    
wind = MLKit()