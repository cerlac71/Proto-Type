"""
Data Ingestion Module for Flood Prediction System (SIH26071)

This module handles:
1. Generation of synthetic rainfall and river gauge data for demo purposes
2. Loading CSV files with historical data
3. Cleaning missing values and preprocessing
4. Outputting a clean pandas DataFrame ready for ML model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import os


def generate_synthetic_data(
    num_stations: int = 5,
    days: int = 365,
    hours_per_day: int = 24,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic historical rainfall and river-gauge readings.
    
    This simulates IMD (India Meteorological Department) and CWC (Central Water Commission)
    data for hackathon demonstration when live APIs are not available.
    
    Args:
        num_stations: Number of weather/gauge stations to simulate
        days: Number of days of historical data
        hours_per_day: Readings per day (hourly data)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: timestamp, station_id, rainfall_mm, river_level_m
    """
    np.random.seed(seed)
    
    # Generate timestamps (hourly readings)
    total_hours = days * hours_per_day
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=total_hours)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')[:-1]  # Exclude end to match total_hours
    
    # Station IDs (simulating different locations in flood-prone areas)
    station_ids = [f"STN_{i:03d}" for i in range(1, num_stations + 1)]
    
    # Generate base data for each station
    data_records = []
    
    for station_id in station_ids:
        # Station-specific parameters (different geography = different patterns)
        base_rainfall = np.random.uniform(0, 5)  # Base rainfall mm/hour (lower baseline)
        base_river_level = np.random.uniform(3, 6)  # Base river level meters (lower baseline)
        
        # Seasonal pattern (monsoon simulation)
        day_indices = np.arange(len(timestamps)) // hours_per_day
        seasonal_factor = np.sin(2 * np.pi * day_indices / 365)  # Annual cycle
        
        # Monsoon amplification (June-September in Indian context)
        monsoon_months = [5, 6, 7, 8]  # June, July, August, September (0-indexed)
        month_indices = (timestamps.month - 1) % 12
        monsoon_mask = np.isin(month_indices, monsoon_months)
        monsoon_factor = np.where(monsoon_mask, 2.0, 1.0)  # Reduced from 2.5 to 2.0
        
        # Generate rainfall with patterns and noise (more varied distribution)
        rainfall = (
            base_rainfall 
            + seasonal_factor * 3  # Reduced from 5 to 3
            + np.random.exponential(scale=2, size=len(timestamps))  # Reduced scale
        ) * monsoon_factor
        
        # Add some extreme rainfall events (cyclones/heavy storms) - more varied
        extreme_events = np.random.choice(
            len(timestamps), 
            size=int(len(timestamps) * 0.03),  # 3% extreme events
            replace=False
        )
        rainfall[extreme_events] *= np.random.uniform(2, 8, size=len(extreme_events))  # Wider range
        
        # Ensure non-negative rainfall
        rainfall = np.maximum(rainfall, 0)
        
        # River level responds to rainfall with lag and accumulation
        # Simple hydrological model: river level = base + accumulated rainfall effect
        rainfall_accumulated = pd.Series(rainfall).rolling(window=24).mean().values  # 24-hour rolling avg
        river_level = (
            base_river_level 
            + rainfall_accumulated * 0.2  # Reduced from 0.3 to 0.2
            + seasonal_factor * 1  # Reduced from 2 to 1
            + np.random.normal(0, 0.3, len(timestamps))  # Reduced noise
        )
        
        # Add some correlation with upstream stations (simplified)
        if station_id != "STN_001":
            river_level += np.random.uniform(0.5, 2.0)  # Downstream stations have higher levels
        
        # Create records for this station
        for i in range(len(timestamps)):
            data_records.append({
                'timestamp': timestamps[i],
                'station_id': station_id,
                'rainfall_mm': round(rainfall[i], 2),
                'river_level_m': round(river_level[i], 2)
            })
    
    df = pd.DataFrame(data_records)
    
    # Introduce some missing values (real-world scenario)
    missing_indices = np.random.choice(
        len(df), 
        size=int(len(df) * 0.03),  # 3% missing data
        replace=False
    )
    
    # Randomly decide which column to make missing for each index
    for idx in missing_indices:
        col_choice = np.random.choice(['rainfall_mm', 'river_level_m'])
        df.loc[idx, col_choice] = np.nan
    
    return df


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load historical rainfall and river-gauge data from a CSV file.
    
    Expected CSV columns:
    - timestamp: ISO format datetime string or parseable date
    - station_id: String identifier for the station
    - rainfall_mm: Numeric rainfall in millimeters
    - river_level_m: Numeric river level in meters
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with parsed and validated data
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_columns = ['timestamp', 'station_id', 'rainfall_mm', 'river_level_m']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Ensure numeric types for measurements
    df['rainfall_mm'] = pd.to_numeric(df['rainfall_mm'], errors='coerce')
    df['river_level_m'] = pd.to_numeric(df['river_level_m'], errors='coerce')
    
    # Convert station_id to string
    df['station_id'] = df['station_id'].astype(str)
    
    return df


def clean_data(df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and invalid data.
    
    Args:
        df: Input DataFrame with raw data
        strategy: Method for handling missing values
            - 'interpolate': Linear interpolation (good for time series)
            - 'forward_fill': Forward fill then backward fill
            - 'drop': Drop rows with any missing values
            - 'mean': Fill with mean value per station
            
    Returns:
        Cleaned DataFrame with no missing values
        
    Raises:
        ValueError: If invalid strategy is provided
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Sort by station and timestamp for proper interpolation
    df_clean = df_clean.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
    
    # Remove rows with invalid timestamps
    df_clean = df_clean.dropna(subset=['timestamp'])
    
    # Handle negative rainfall (invalid)
    df_clean.loc[df_clean['rainfall_mm'] < 0, 'rainfall_mm'] = 0
    
    # Handle negative river levels (invalid)
    invalid_river = df_clean['river_level_m'] < 0
    if invalid_river.any():
        median_level = df_clean.loc[~invalid_river, 'river_level_m'].median()
        df_clean.loc[invalid_river, 'river_level_m'] = median_level
    
    # Apply missing value strategy
    if strategy == 'interpolate':
        # Interpolate within each station group
        numeric_cols = ['rainfall_mm', 'river_level_m']
        for col in numeric_cols:
            df_clean[col] = df_clean.groupby('station_id')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        # Fill any remaining NaNs at edges
        for col in numeric_cols:
            df_clean[col] = df_clean.groupby('station_id')[col].transform(
                lambda x: x.bfill().ffill()
            )
        
    elif strategy == 'forward_fill':
        for col in ['rainfall_mm', 'river_level_m']:
            df_clean[col] = df_clean.groupby('station_id')[col].transform(
                lambda x: x.ffill().bfill()
            )
        
    elif strategy == 'drop':
        df_clean = df_clean.dropna(subset=['rainfall_mm', 'river_level_m'])
        
    elif strategy == 'mean':
        # Fill with station-wise mean
        for col in ['rainfall_mm', 'river_level_m']:
            station_means = df_clean.groupby('station_id')[col].transform('mean')
            df_clean[col] = df_clean[col].fillna(station_means)
        # Fill any remaining NaNs (stations with all missing)
        df_clean = df_clean.fillna(df_clean[['rainfall_mm', 'river_level_m']].mean())
        
    else:
        raise ValueError(f"Invalid cleaning strategy: {strategy}. Choose from: interpolate, forward_fill, drop, mean")
    
    # Final check - drop any remaining rows with NaN in critical columns
    df_clean = df_clean.dropna(subset=['rainfall_mm', 'river_level_m'])
    
    return df_clean


def ingest_data(
    source: str = 'synthetic',
    file_path: Optional[str] = None,
    clean_strategy: str = 'interpolate',
    **kwargs
) -> pd.DataFrame:
    """
    Main ingestion function that loads and cleans data from various sources.
    
    Args:
        source: Data source type
            - 'synthetic': Generate synthetic demo data
            - 'csv': Load from CSV file
        file_path: Path to CSV file (required if source='csv')
        clean_strategy: Strategy for cleaning missing values
        **kwargs: Additional arguments passed to data generation (for synthetic)
        
    Returns:
        Clean pandas DataFrame ready for model training
        
    Example:
        # Generate synthetic data
        df = ingest_data(source='synthetic', num_stations=5, days=30)
        
        # Load from CSV
        df = ingest_data(source='csv', file_path='data/historical.csv')
    """
    if source == 'synthetic':
        print("Generating synthetic rainfall and river-gauge data...")
        df = generate_synthetic_data(**kwargs)
        
    elif source == 'csv':
        if file_path is None:
            raise ValueError("file_path is required when source='csv'")
        print(f"Loading data from CSV: {file_path}")
        df = load_csv_data(file_path)
        
    else:
        raise ValueError(f"Unknown data source: {source}. Choose from: synthetic, csv")
    
    print(f"Loaded {len(df)} records from {df['station_id'].nunique()} stations")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check for missing values before cleaning
    missing_before = df[['rainfall_mm', 'river_level_m']].isnull().sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values. Cleaning with strategy: {clean_strategy}")
    
    # Clean the data
    df_clean = clean_data(df, strategy=clean_strategy)
    
    # Verify no missing values remain
    missing_after = df_clean[['rainfall_mm', 'river_level_m']].isnull().sum().sum()
    if missing_after > 0:
        raise ValueError(f"Cleaning failed: {missing_after} missing values still remain")
    
    print(f"Data cleaning complete. Final dataset: {len(df_clean)} records")
    
    return df_clean


def save_sample_dataset(output_path: str = 'ml-pipeline/data/sample_data.csv'):
    """
    Generate and save a sample dataset for testing and demonstration.
    
    Args:
        output_path: Path where the sample CSV will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate 30 days of data for 5 stations (manageable size for demo)
    df = generate_synthetic_data(num_stations=5, days=30, seed=42)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to: {output_path}")
    print(f"Records: {len(df)}, Stations: {df['station_id'].nunique()}")
    
    return df


if __name__ == "__main__":
    # Demo: Generate sample dataset and process it
    print("=" * 60)
    print("Data Ingestion Module - Demo")
    print("=" * 60)
    
    # Generate and save sample data
    sample_df = save_sample_dataset()
    
    print("\n" + "=" * 60)
    print("Loading and cleaning the sample data...")
    print("=" * 60)
    
    # Load and clean using the main ingestion function
    cleaned_df = ingest_data(source='csv', file_path='ml-pipeline/data/sample_data.csv')
    
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"\nShape: {cleaned_df.shape}")
    print(f"\nColumns: {list(cleaned_df.columns)}")
    print(f"\nData types:\n{cleaned_df.dtypes}")
    print(f"\nStatistics:\n{cleaned_df.describe()}")
    print(f"\nSample records:\n{cleaned_df.head(10)}")
    
    print("\n✓ Data ingestion complete! DataFrame ready for model training.")
