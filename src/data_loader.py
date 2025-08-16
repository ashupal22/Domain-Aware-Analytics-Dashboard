"""
Centralized data loading following Single Responsibility Principle
Handles all CSV reading, encoding detection, and preprocessing
"""

import pandas as pd
import numpy as np
import os
import json
import io
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CSVDataLoader:
    """Handles all CSV data loading and preprocessing"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.errors: list = []
    
    def detect_delimiter(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Detect CSV delimiter"""
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = [f.readline() for _ in range(5) if f.readline()]
            
            delimiters = [',', '\t', ';', '|']
            delimiter_counts = {}
            
            for delim in delimiters:
                counts = [line.count(delim) for line in lines]
                if len(set(counts)) == 1 and counts[0] > 0:
                    delimiter_counts[delim] = counts[0]
            
            return max(delimiter_counts, key=delimiter_counts.get) if delimiter_counts else ','
        except:
            return ','
    
    def load_csv(self, file_path: str, save_processed: bool = True) -> Tuple[bool, str]:
        """
        Load CSV with comprehensive error handling and preprocessing
        Returns: (success, message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                delimiter = self.detect_delimiter(file_path, encoding)
                
                # Try multiple reading strategies
                success = self._try_reading_strategies(file_path, encoding, delimiter)
                if success:
                    break
                    
            except Exception as e:
                self.errors.append(f"Encoding {encoding}: {str(e)}")
                continue
        
        if self.df is None:
            return False, f"Failed to read CSV: {'; '.join(self.errors)}"
        
        # Store metadata
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.metadata = {
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'encoding_used': encoding,
            'delimiter': delimiter,
            'original_shape': self.df.shape,
            'load_timestamp': datetime.now().isoformat(),
            'errors_found': len(self.errors)
        }
        
        # Preprocess data
        self._preprocess_data()
        
        # Save processed data if requested
        if save_processed:
            self._save_processed_data()
        
        return True, f"Successfully loaded {len(self.df):,} rows, {len(self.df.columns)} columns"
    
    def _try_reading_strategies(self, file_path: str, encoding: str, delimiter: str) -> bool:
        """Try different pandas reading strategies"""
        strategies = [
            # Strategy 1: Standard read
            lambda: pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, 
                               on_bad_lines='skip', engine='python'),
            
            # Strategy 2: Legacy pandas
            lambda: pd.read_csv(file_path, encoding=encoding, delimiter=delimiter,
                               error_bad_lines=False, warn_bad_lines=True, engine='python'),
            
            # Strategy 3: Manual line processing
            lambda: self._manual_line_processing(file_path, encoding, delimiter)
        ]
        
        for strategy in strategies:
            try:
                result = strategy()
                if result is not None and not result.empty:
                    self.df = result
                    return True
            except:
                continue
        
        return False
    
    def _manual_line_processing(self, file_path: str, encoding: str, delimiter: str) -> pd.DataFrame:
        """Manual line-by-line processing for problematic files"""
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
        
        valid_lines = []
        expected_cols = None
        
        for i, line in enumerate(lines):
            parts = line.strip().split(delimiter)
            
            if i == 0:
                expected_cols = len(parts)
                valid_lines.append(line)
            elif len(parts) == expected_cols:
                valid_lines.append(line)
            else:
                self.errors.append(f"Line {i+1}: Expected {expected_cols} columns, found {len(parts)}")
        
        if valid_lines:
            return pd.read_csv(io.StringIO(''.join(valid_lines)), 
                             delimiter=delimiter, encoding=encoding)
        return pd.DataFrame()
    
    def _preprocess_data(self):
        """Intelligent data preprocessing"""
        if self.df is None:
            return
        
        # Clean column names (same as profiler)
        original_columns = self.df.columns.tolist()
        self.df.columns = [str(col).strip().replace(' ', '').replace('.', '').lower() 
                          for col in self.df.columns]
        
        # Store column mapping
        self.metadata['column_mapping'] = dict(zip(self.df.columns, original_columns))
        
        # Convert data types intelligently
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Try datetime conversion
                try:
                    sample = self.df[col].dropna().head(100)
                    if (sample.str.match(r'\d{4}-\d{2}-\d{2}').any() or 
                        sample.str.match(r'\d{2}/\d{2}/\d{4}').any()):
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        continue
                except:
                    pass
                
                # Try numeric conversion
                try:
                    if (self.df[col].str.replace(',', '').str.replace('$', '')
                        .str.replace('%', '').str.replace(' ', '').str.isnumeric().any()):
                        self.df[col] = pd.to_numeric(
                            self.df[col].astype(str).str.replace(',', '')
                            .str.replace('$', '').str.replace('%', ''), 
                            errors='ignore'
                        )
                except:
                    pass
        
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Update metadata
        self.metadata['processed_shape'] = self.df.shape
        self.metadata['dtypes'] = self.df.dtypes.to_dict()
    
    def _save_processed_data(self):
        """Save processed data and metadata"""
        try:
            os.makedirs('data/processed', exist_ok=True)
            
            # Save processed CSV
            processed_file = 'data/processed/data_cleaned.csv'
            self.df.to_csv(processed_file, index=False)
            
            # Save metadata
            metadata_file = 'data/processed/metadata.json'
            with open(metadata_file, 'w') as f:
                # Convert non-serializable types
                serializable_metadata = {}
                for key, value in self.metadata.items():
                    if key == 'dtypes':
                        serializable_metadata[key] = {k: str(v) for k, v in value.items()}
                    else:
                        serializable_metadata[key] = value
                
                json.dump(serializable_metadata, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not save processed data: {e}")
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded and processed DataFrame"""
        return self.df.copy() if self.df is not None else pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get loading and processing metadata"""
        return self.metadata.copy()
    
    def load_processed_data(self, processed_file: str = 'data/processed/data_cleaned.csv') -> bool:
        """Load previously processed data"""
        try:
            if os.path.exists(processed_file):
                self.df = pd.read_csv(processed_file)
                
                # Load metadata if available
                metadata_file = 'data/processed/metadata.json'
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        self.metadata = json.load(f)
                
                return True
        except Exception as e:
            print(f"⚠️ Could not load processed data: {e}")
        
        return False