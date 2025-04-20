import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler

def select_file():
    # root = tk.Tk()
    # root.withdraw()
    # file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    # return file_path
    return "landmarks.csv"

# Select CSV file
csv_file = select_file()
if not csv_file:
    print("No file selected. Exiting.")
    exit()

# Load CSV
print(f"Loading {csv_file}...")
df = pd.read_csv(csv_file)

# Separate labels and numeric data
label_column = "Label"
numeric_columns = [col for col in df.columns if col != label_column]

# Normalize numerical data
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save normalized data
df.to_csv("normalized_landmarks.csv", index=False)
print("Normalization complete. Data saved to normalized_landmarks.csv")