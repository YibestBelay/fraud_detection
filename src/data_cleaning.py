"""
Data cleaning module for fraud detection.
Implements comprehensive data cleaning with justifications.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from typing import Tuple, Optional, Dict, Any


class FraudDataCleaner:
    """
    Comprehensive data cleaner for fraud detection datasets.
    Handles missing values, duplicates, and data type corrections.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize data cleaner.
        
        Args:
            verbose (bool): Whether to print cleaning steps
        """
        self.verbose = verbose
        self.cleaning_report = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset with proper error handling.
        
        Args:
            filepath (str): Path to data file
            
        Returns:
            pandas.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is incorrect
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                raise ValueError("Unsupported file format. Use .csv or .parquet")
                
            if self.verbose:
                print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
                print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
            return df
            
        except FileNotFoundError:
            print(f"✗ Error: File not found at {filepath}")
            raise
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Missing values analysis
        """
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'missing_per_column': df.isnull().sum(),
            'missing_percentage': df.isnull().mean() * 100
        }
        
        if self.verbose:
            print("\n" + "="*60)
            print("MISSING VALUES ANALYSIS")
            print("="*60)
            missing_cols = missing_stats['missing_percentage'][missing_stats['missing_percentage'] > 0]
            if len(missing_cols) > 0:
                print(f"Total missing values: {missing_stats['total_missing']:,}")
                print("\nColumns with missing values:")
                for col, pct in missing_cols.items():
                    print(f"  {col}: {pct:.2f}% missing")
            else:
                print("✓ No missing values found")
                
        return missing_stats
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'median',
                             threshold: float = 0.3) -> pd.DataFrame:
        """
        Handle missing values with justification.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): 'median', 'mean', 'mode', or 'drop'
            threshold (float): Drop columns with > threshold% missing
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        initial_rows = df_clean.shape[0]
        initial_cols = df_clean.shape[1]
        
        # Strategy justification documentation
        strategy_justification = {
            'median': "Robust to outliers, suitable for skewed distributions",
            'mean': "Preserves mean of distribution, good for normal distributions",
            'mode': "For categorical variables, most frequent value",
            'drop': "When missing data is minimal or imputation would introduce bias"
        }
        
        if self.verbose:
            print("\n" + "="*60)
            print(f"HANDLING MISSING VALUES (Strategy: {strategy})")
            print("="*60)
            print(f"Justification: {strategy_justification.get(strategy, 'Custom strategy')}")
        
        # Drop columns with high missing percentage
        missing_pct = df_clean.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            if self.verbose:
                print(f"✗ Dropped columns with >{threshold*100}% missing: {cols_to_drop}")
        
        # Handle remaining missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    if strategy == 'median':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif strategy == 'mean':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif strategy == 'mode':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    # For categorical/object columns, use mode
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        if self.verbose:
            print(f"✓ Missing values handled")
            print(f"  Columns before: {initial_cols}, after: {df_clean.shape[1]}")
            print(f"  Rows preserved: {initial_rows:,}")
            print(f"  Remaining missing values: {df_clean.isnull().sum().sum()}")
        
        self.cleaning_report['missing_values'] = {
            'strategy': strategy,
            'justification': strategy_justification.get(strategy),
            'dropped_columns': cols_to_drop,
            'rows_preserved': initial_rows
        }
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate rows with justification.
        
        Args:
            df (pd.DataFrame): Input dataframe
            subset (list, optional): Columns to check for duplicates
            
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        initial_rows = df.shape[0]
        
        if self.verbose:
            print("\n" + "="*60)
            print("REMOVING DUPLICATES")
            print("="*60)
            print("Justification: Duplicate transactions may indicate data collection errors")
        
        # Identify duplicates
        if subset:
            duplicates = df.duplicated(subset=subset, keep='first')
        else:
            duplicates = df.duplicated(keep='first')
        
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            df_clean = df[~duplicates].reset_index(drop=True)
            if self.verbose:
                print(f"✗ Removed {duplicate_count:,} duplicate rows")
                print(f"✓ Rows before: {initial_rows:,}, after: {df_clean.shape[0]:,}")
                print(f"✓ Duplicate percentage: {duplicate_count/initial_rows*100:.2f}%")
        else:
            df_clean = df.copy()
            if self.verbose:
                print("✓ No duplicates found")
        
        self.cleaning_report['duplicates'] = {
            'removed_count': duplicate_count,
            'removed_percentage': duplicate_count/initial_rows*100 if initial_rows > 0 else 0
        }
        
        return df_clean
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct data types with optimization.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with corrected types
        """
        df_clean = df.copy()
        
        if self.verbose:
            print("\n" + "="*60)
            print("CORRECTING DATA TYPES")
            print("="*60)
            print("Justification: Optimize memory usage and ensure proper analysis")
        
        # Dictionary of common type corrections
        type_corrections = {}
        
        for col in df_clean.columns:
            col_dtype = str(df_clean[col].dtype)
            
            # Convert object to categorical if low cardinality
            if col_dtype == 'object':
                unique_count = df_clean[col].nunique()
                if unique_count < 50 and unique_count < len(df_clean) * 0.1:
                    df_clean[col] = df_clean[col].astype('category')
                    type_corrections[col] = f'object -> category ({unique_count} unique values)'
            
            # Downcast numerical columns
            elif col_dtype.startswith('int'):
                col_min = df_clean[col].min()
                col_max = df_clean[col].max()
                
                if col_min >= 0:
                    if col_max < 255:
                        df_clean[col] = pd.to_numeric(df_clean[col], downcast='unsigned')
                    elif col_max < 65535:
                        df_clean[col] = pd.to_numeric(df_clean[col], downcast='unsigned')
                else:
                    if col_min > -128 and col_max < 127:
                        df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')
            
            # Convert string dates to datetime
            elif col_dtype == 'object' and any(keyword in col.lower() 
                                             for keyword in ['date', 'time', 'timestamp']):
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    type_corrections[col] = 'object -> datetime'
                except:
                    pass
        
        if self.verbose and type_corrections:
            print("Type corrections applied:")
            for col, correction in type_corrections.items():
                print(f"  {col}: {correction}")
        elif self.verbose:
            print("✓ Data types already optimal")
        
        # Memory optimization report
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_memory = df_clean.memory_usage(deep=True).sum() / 1024**2
        memory_saved = original_memory - optimized_memory
        
        if self.verbose:
            print(f"✓ Memory usage: {original_memory:.2f}MB → {optimized_memory:.2f}MB")
            print(f"✓ Memory saved: {memory_saved:.2f}MB ({memory_saved/original_memory*100:.1f}%)")
        
        self.cleaning_report['data_types'] = {
            'type_corrections': type_corrections,
            'memory_saved_mb': memory_saved,
            'memory_saved_percentage': memory_saved/original_memory*100 if original_memory > 0 else 0
        }
        
        return df_clean
    
    def clean_fraud_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive cleaning for fraud dataset.
        
        Args:
            df (pd.DataFrame): Raw fraud data
            
        Returns:
            pd.DataFrame: Cleaned fraud data
        """
        if self.verbose:
            print("\n" + "="*80)
            print("COMPREHENSIVE DATA CLEANING - FRAUD DATASET")
            print("="*80)
        
        # Step 1: Analyze missing values
        missing_stats = self.analyze_missing_values(df)
        
        # Step 2: Handle missing values (using median for numerical, mode for categorical)
        df_clean = self.handle_missing_values(df, strategy='median')
        
        # Step 3: Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # Step 4: Correct data types
        df_clean = self.correct_data_types(df_clean)
        
        # Step 5: Validate cleaning
        self.validate_cleaning(df, df_clean)
        
        return df_clean
    
    def validate_cleaning(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        """
        Validate cleaning process.
        
        Args:
            df_before (pd.DataFrame): Original dataframe
            df_after (pd.DataFrame): Cleaned dataframe
        """
        if self.verbose:
            print("\n" + "="*60)
            print("CLEANING VALIDATION")
            print("="*60)
            
            # Check for remaining issues
            issues = []
            
            # Missing values
            if df_after.isnull().sum().sum() > 0:
                issues.append(f"Missing values: {df_after.isnull().sum().sum()}")
            
            # Duplicates
            if df_after.duplicated().sum() > 0:
                issues.append(f"Duplicates: {df_after.duplicated().sum()}")
            
            if issues:
                print("✗ Issues remaining:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✓ All issues resolved")
            
            # Summary
            print("\n" + "-"*40)
            print("CLEANING SUMMARY")
            print("-"*40)
            print(f"Rows: {df_before.shape[0]:,} → {df_after.shape[0]:,}")
            print(f"Columns: {df_before.shape[1]} → {df_after.shape[1]}")
            print(f"Memory: {df_before.memory_usage(deep=True).sum()/1024**2:.2f}MB → "
                  f"{df_after.memory_usage(deep=True).sum()/1024**2:.2f}MB")
    
    def generate_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning report.
        
        Returns:
            dict: Cleaning report with all metrics
        """
        return self.cleaning_report