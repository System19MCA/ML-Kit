from tkinter import filedialog
import os
import tkinter as tk
from tkinter import ttk

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
