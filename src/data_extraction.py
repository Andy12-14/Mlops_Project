import pandas as pd
import os


def load_data(file_path):
    """
    Load raw data from a csv file. 
    Handle missing flies and wrong data formats.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try to load the CSV file
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"File is not in valid CSV format: {e}")
    
    return data