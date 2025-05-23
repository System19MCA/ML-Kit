from tkinter import filedialog
import os
import tkinter as tk
from tkinter import ttk
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

def load_csv_file():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), 
                                        title="Select a File",
                                        filetypes=(("csv", "*.*"),))
    return filename


def showDF(frame, df, title="Results"):
    x = tk.Toplevel(frame)
        
    x.title(title)

        # Create a frame for the table and scrollbars
    table_frame = ttk.Frame(x)
    table_frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbars
    x_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
    y_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)

    # Create the Treeview widget (table)
    tree = ttk.Treeview(table_frame,
                            columns=list(df.columns),
                            show='headings',
                            xscrollcommand=x_scrollbar.set,
                            yscrollcommand=y_scrollbar.set)

        # Configure scrollbars to work with the Treeview
    x_scrollbar.config(command=tree.xview)
    y_scrollbar.config(command=tree.yview)

        # Add scrollbars to the frame
    x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Define column headings
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # Adjust column width as needed

        # Insert data into the Treeview
    for index, row in df.iterrows():
        tree.insert('', tk.END, values=list(row))

        # Pack the Treeview
    tree.pack(fill=tk.BOTH, expand=True)

    x.mainloop()    


def plot_classification_report(y_true, y_pred, target_names=None):
    """
    Plots the classification report as a heatmap.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        target_names (list, optional): List of target class names. Defaults to None.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    fig, ax = plt.subplots(figsize=(8, len(df)/2))
    sns.heatmap(df.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)

    plt.title("Classification Report")
    plt.ylabel("Metrics")
    plt.xlabel("Classes")
    plt.show()