"""
Data profiling following Single Responsibility Principle
Depends on DataLoader for data access
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any
from datetime import datetime

class DataProfiler:
    """Handles data profiling and analysis"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.profiles: List[Dict] = []
        self.summary: Dict[str, Any] = {}
    
    def infer_column_type(self, series: pd.Series) -> str:
        """Infer the actual type of a column"""
        if series.isna().all():
            return 'empty'
        
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'empty'
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            if all(non_null == non_null.astype(int)):
                return 'integer'
            else:
                return 'float'
        
        # Sample for checking
        sample = non_null.head(100) if len(non_null) > 100 else non_null
        
        # Check for boolean
        bool_values = {'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0'}
        unique_lower = set(str(v).lower().strip() for v in sample.unique())
        if len(unique_lower) <= 3 and unique_lower.issubset(bool_values):
            return 'boolean'
        
        # Check for datetime
        try:
            pd.to_datetime(sample, errors='raise')
            return 'datetime'
        except:
            pass
        
        # Check if can be numeric
        try:
            cleaned = sample.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '').str.strip()
            pd.to_numeric(cleaned, errors='raise')
            return 'numeric_string'
        except:
            pass
        
        # Distinguish between categorical and text
        unique_ratio = len(non_null.unique()) / len(non_null)
        avg_length = non_null.astype(str).str.len().mean()
        
        if unique_ratio < 0.5 and avg_length < 50:
            return 'categorical'
        else:
            return 'text'
    
    def profile_dataset(self) -> bool:
        """Generate comprehensive profile of the dataset"""
        df = self.data_loader.get_data()
        if df.empty:
            return False
        
        self.profiles = []
        
        # Profile each column
        for col in df.columns:
            try:
                profile = self._profile_single_column(df, col)
                self.profiles.append(profile)
            except Exception as e:
                print(f"⚠️ Error profiling column {col}: {e}")
                self.profiles.append({
                    'name': col,
                    'type': 'unknown',
                    'nulls': 0,
                    'null_pct': 0,
                    'unique': 0,
                    'cardinality': 'unknown'
                })
        
        # Generate summary
        self._generate_summary(df)
        return True
    
    def _profile_single_column(self, df: pd.DataFrame, col: str) -> Dict:
        """Profile a single column"""
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        col_type = self.infer_column_type(df[col])
        
        # Determine cardinality
        if unique_count <= 10:
            cardinality = "low"
        elif unique_count <= 100:
            cardinality = "medium"
        else:
            cardinality = "high"
        
        profile = {
            'name': col,
            'type': col_type,
            'nulls': int(null_count),
            'null_pct': null_pct,
            'unique': int(unique_count),
            'cardinality': cardinality
        }
        
        # Add type-specific information
        if col_type in ['integer', 'float']:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                profile["range"] = [float(non_null.min()), float(non_null.max())]
        elif col_type == 'categorical':
            top_values = df[col].value_counts().head(3)
            profile["top_values"] = [str(v) for v in top_values.index]
        
        return profile
    
    def _generate_summary(self, df: pd.DataFrame):
        """Generate summary for LLM processing"""
        self.summary = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": self.profiles,
            "sample_rows": [],
            "metadata": self.data_loader.get_metadata()
        }
        
        # Add sample rows
        try:
            sample_df = df.head(3).fillna('')
            for _, row in sample_df.iterrows():
                row_dict = {}
                for col in df.columns:
                    try:
                        val = row[col]
                        if pd.isna(val):
                            row_dict[col] = None
                        elif isinstance(val, (int, float, str, bool)):
                            row_dict[col] = val
                        else:
                            row_dict[col] = str(val)
                    except:
                        row_dict[col] = ""
                self.summary["sample_rows"].append(row_dict)
        except Exception as e:
            print(f"⚠️ Could not serialize sample rows: {e}")
    
    def save_profile(self, output_file: str) -> bool:
        """Save profile to JSON file"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(self.summary, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"❌ Error saving profile: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get the profile summary"""
        return self.summary.copy()