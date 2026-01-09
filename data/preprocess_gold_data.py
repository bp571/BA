#!/usr/bin/env python3
"""
Preprocess gold data CSV to make it compatible with rolling forecast scripts.
The original gold data has multiple header rows and needs to be reformatted.
"""

import pandas as pd
import os

def preprocess_gold_data(input_path, output_path):
    """
    Preprocess gold data CSV to standard format expected by rolling forecast scripts
    
    Args:
        input_path: Path to original gold data CSV
        output_path: Path to save preprocessed CSV
    """
    # Read the CSV file, skipping the first 3 rows which contain headers/metadata
    # Row 0: Price,Close,High,Low,Open,Volume
    # Row 1: Ticker,GC=F,GC=F,GC=F,GC=F,GC=F  
    # Row 2: Datetime,,,,,
    # Row 3+: Actual data
    
    print(f"Reading gold data from: {input_path}")
    
    # Read the raw file to understand structure
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    print(f"First few lines of original file:")
    for i, line in enumerate(lines[:5]):
        print(f"Row {i}: {line.strip()}")
    
    # Read data starting from row 3 (0-indexed), and manually assign column names
    df = pd.read_csv(input_path, skiprows=3, header=None)
    
    # Assign proper column names based on the first row structure
    df.columns = ['datetime', 'close', 'high', 'low', 'open', 'volume']
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    # Convert datetime column to proper datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by datetime to ensure chronological order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Select only the columns needed for forecasting
    # Keep all OHLCV columns for flexibility, but datetime and close are required
    df_processed = df[['datetime', 'close', 'high', 'low', 'open', 'volume']].copy()
    
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Processed columns: {df_processed.columns.tolist()}")
    print(f"Date range: {df_processed['datetime'].min()} to {df_processed['datetime'].max()}")
    print(f"Sample of processed data:")
    print(df_processed.head())
    print(f"Data types:")
    print(df_processed.dtypes)
    
    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")
    
    return df_processed

if __name__ == "__main__":
    input_file = "data/processed/gold_2025.csv"
    output_file = "data/processed/gold_2025_processed.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    
    try:
        preprocessed_df = preprocess_gold_data(input_file, output_file)
        print(f"✅ Successfully preprocessed gold data!")
        print(f"📁 Preprocessed file: {output_file}")
        print(f"📊 Records: {len(preprocessed_df)}")
        
    except Exception as e:
        print(f"❌ Error preprocessing gold data: {e}")
        import traceback
        traceback.print_exc()