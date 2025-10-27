import pytest
import pandas as pd
import os
from data_extraction import load_data


def test_load_data_creates_dataframe_with_expected_columns():
    """Test that load_data returns a DataFrame with the expected columns."""
    file_path = "dataset/dataset.csv"
    df = load_data(file_path)
    
    expected_columns = [
        'reviewId', 'userName', 'userImage', 'content', 'score', 
        'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 
        'repliedAt', 'sortOrder', 'appId'
    ]
    
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns


def test_load_data_raises_error_for_missing_file():
    """Test that load_data raises FileNotFoundError when file is missing."""
    file_path = "nonexistent_file.csv"
    
    with pytest.raises(FileNotFoundError):
        load_data(file_path)


def test_load_data_raises_error_for_invalid_csv_format():
    """Test that load_data raises ValueError when file is not in CSV format."""
    # Create a temporary binary file that cannot be parsed as CSV
    invalid_file_path = "tests/unit/test_invalid_file.bin"
    with open(invalid_file_path, 'wb') as f:
        f.write(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A')  # Binary data (PNG header)
    
    try:
        with pytest.raises(ValueError):
            load_data(invalid_file_path)
    finally:
        # Clean up the temporary file
        if os.path.exists(invalid_file_path):
            os.remove(invalid_file_path)

