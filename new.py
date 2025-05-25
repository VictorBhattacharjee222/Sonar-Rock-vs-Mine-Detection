import pandas as pd
import os

# Check if file exists
filepath = "data/sonar.csv"
print(f"File exists: {os.path.exists(filepath)}")

if os.path.exists(filepath):
    try:
        # Try to read the CSV
        df = pd.read_csv(filepath)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nLast column unique values:")
        print(df.iloc[:, -1].unique())
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Try reading first few lines as text
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"Line {i}: {line.strip()}")
else:
    print("CSV file not found!")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    if os.path.exists('data'):
        print("Files in data directory:", os.listdir('data'))