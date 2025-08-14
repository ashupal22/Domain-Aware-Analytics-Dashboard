#!/usr/bin/env python3
"""
Simple CSV Profiler Script - Robust Version
Usage: python profiler.py <csv_file_path>
Handles problematic CSVs with encoding issues, inconsistent columns, etc.
"""

import pandas as pd
import numpy as np
import sys
import json
from datetime import datetime
import warnings
import io
warnings.filterwarnings('ignore')

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color):
    """Print colored text in terminal"""
    print(f"{color}{text}{Colors.ENDC}")

def print_separator():
    """Print a separator line"""
    print("=" * 80)

def detect_delimiter(file_path, encoding='utf-8'):
    """Detect the delimiter used in CSV"""
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            # Read first few lines
            lines = []
            for i in range(5):
                line = f.readline()
                if line:
                    lines.append(line)
            
            # Count delimiters
            delimiters = [',', '\t', ';', '|']
            delimiter_counts = {}
            
            for delim in delimiters:
                counts = [line.count(delim) for line in lines]
                # Check consistency
                if len(set(counts)) == 1 and counts[0] > 0:
                    delimiter_counts[delim] = counts[0]
            
            if delimiter_counts:
                # Return delimiter with most consistent count
                return max(delimiter_counts, key=delimiter_counts.get)
    except:
        pass
    
    return ','  # Default to comma

def read_csv_smart(file_path):
    """Read CSV with automatic encoding and error handling"""
    print_colored("\n📂 STEP 1: READING CSV FILE", Colors.HEADER)
    print_separator()
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    df = None
    used_encoding = None
    errors_found = []
    
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}...", end=" ")
            
            # First, try to detect delimiter
            delimiter = detect_delimiter(file_path, encoding)
            print(f"(delimiter: '{delimiter}')", end=" ")
            
            # Try reading with various error handling strategies
            try:
                # Method 1: Standard read
                df = pd.read_csv(file_path, 
                               encoding=encoding, 
                               delimiter=delimiter,
                               on_bad_lines='skip',  # Skip bad lines
                               engine='python')  # Python engine is more flexible
                used_encoding = encoding
                print_colored("✓ Success!", Colors.GREEN)
                break
                
            except Exception as e1:
                # Method 2: Try with error_bad_lines for older pandas
                try:
                    df = pd.read_csv(file_path, 
                                   encoding=encoding,
                                   delimiter=delimiter,
                                   error_bad_lines=False,  # Skip bad lines (older pandas)
                                   warn_bad_lines=True,
                                   engine='python')
                    used_encoding = encoding
                    print_colored("✓ Success (with warnings)!", Colors.YELLOW)
                    break
                    
                except Exception as e2:
                    # Method 3: Read line by line and handle errors
                    try:
                        print_colored("✗ Standard read failed, trying line-by-line...", Colors.YELLOW)
                        
                        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                            lines = f.readlines()
                        
                        # Parse manually
                        valid_lines = []
                        header = None
                        expected_cols = None
                        
                        for i, line in enumerate(lines):
                            parts = line.strip().split(delimiter)
                            
                            if i == 0:
                                header = parts
                                expected_cols = len(parts)
                                valid_lines.append(line)
                            else:
                                # Allow lines with correct number of columns
                                if len(parts) == expected_cols:
                                    valid_lines.append(line)
                                else:
                                    errors_found.append(f"Line {i+1}: Expected {expected_cols} columns, found {len(parts)}")
                        
                        # Create dataframe from valid lines
                        if valid_lines:
                            df = pd.read_csv(io.StringIO(''.join(valid_lines)), 
                                           delimiter=delimiter,
                                           encoding=encoding)
                            used_encoding = encoding
                            print_colored(f"✓ Recovered {len(valid_lines)-1} valid rows!", Colors.GREEN)
                            if errors_found[:3]:  # Show first 3 errors
                                print_colored(f"  ⚠️  Skipped {len(errors_found)} bad lines", Colors.YELLOW)
                                for err in errors_found[:3]:
                                    print(f"     - {err}")
                                if len(errors_found) > 3:
                                    print(f"     ... and {len(errors_found)-3} more")
                            break
                        else:
                            raise Exception("No valid lines found")
                            
                    except Exception as e3:
                        print_colored(f"✗ Failed: {str(e3)[:50]}", Colors.RED)
                        
        except Exception as e:
            print_colored(f"✗ Failed: {str(e)[:50]}", Colors.RED)
            continue
    
    if df is None:
        print_colored("ERROR: Could not read file with any method!", Colors.RED)
        print("\nTrying to diagnose the issue...")
        
        # Try to show file preview
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read(500)
                print("\nFirst 500 bytes of file (raw):")
                print(raw_bytes[:500])
        except:
            pass
            
        return None
    
    # Clean column names
    df.columns = [str(col).strip().replace(' ', '').replace('.', '').lower() for col in df.columns]
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    print(f"\n📊 File Statistics:")
    print(f"  • File size: {file_size_mb:.2f} MB")
    print(f"  • Encoding used: {used_encoding}")
    print(f"  • Delimiter: '{delimiter if 'delimiter' in locals() else ','}'")
    print_colored(f"  ✓ Successfully loaded {len(df):,} rows and {len(df.columns)} columns", Colors.GREEN)
    
    if file_size_mb > 15:
        print_colored("  ⚠️  Warning: File size exceeds 15MB limit!", Colors.YELLOW)
    
    # Show if any rows were skipped
    if errors_found:
        print_colored(f"  ⚠️  {len(errors_found)} rows skipped due to format issues", Colors.YELLOW)
    
    return df

def infer_column_type(series):
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
        # Clean common number formats
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

def safe_numeric_stats(series, col_type):
    """Safely compute numeric statistics"""
    try:
        if col_type == 'numeric_string':
            # Clean and convert
            numeric_col = pd.to_numeric(
                series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''), 
                errors='coerce'
            )
        else:
            numeric_col = series
        
        non_null = numeric_col.dropna()
        
        if len(non_null) == 0:
            return None
            
        stats = {
            'min': float(non_null.min()),
            'max': float(non_null.max()),
            'mean': float(non_null.mean()),
            'std': float(non_null.std()) if len(non_null) > 1 else 0,
            'q25': float(non_null.quantile(0.25)),
            'median': float(non_null.median()),
            'q75': float(non_null.quantile(0.75))
        }
        
        # Outliers
        Q1 = stats['q25']
        Q3 = stats['q75']
        IQR = Q3 - Q1
        outliers = non_null[(non_null < Q1 - 1.5*IQR) | (non_null > Q3 + 1.5*IQR)]
        stats['outlier_count'] = len(outliers)
        stats['outlier_pct'] = (len(outliers) / len(non_null)) * 100
        
        return stats
    except:
        return None

def profile_dataset(df):
    """Generate comprehensive profile of the dataset"""
    print_colored("\n📊 STEP 2: DATA PROFILING", Colors.HEADER)
    print_separator()
    
    # Basic info
    print_colored("\n📈 Dataset Overview:", Colors.CYAN)
    print(f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for duplicates
    try:
        dup_count = df.duplicated().sum()
        print(f"  • Duplicated rows: {dup_count:,}")
    except:
        print(f"  • Duplicated rows: Unable to compute")
    
    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        print_colored(f"  ⚠️  Empty columns found: {', '.join(empty_cols)}", Colors.YELLOW)
    
    # Column analysis
    print_colored("\n🔍 Column Analysis:", Colors.CYAN)
    print_separator()
    
    profiles = []
    
    for col_idx, col in enumerate(df.columns):
        print(f"\n📌 [{col_idx+1}/{len(df.columns)}] Column: {Colors.BOLD}{col}{Colors.ENDC}")
        
        try:
            # Basic stats
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Infer type
            col_type = infer_column_type(df[col])
            
            # Cardinality
            if unique_count <= 10:
                cardinality = "LOW"
                card_color = Colors.GREEN
            elif unique_count <= 100:
                cardinality = "MEDIUM"
                card_color = Colors.YELLOW
            else:
                cardinality = "HIGH"
                card_color = Colors.RED
            
            print(f"  Type: {Colors.BLUE}{col_type.upper()}{Colors.ENDC}")
            print(f"  Nulls: {null_count:,} ({null_pct:.1f}%)")
            print(f"  Unique: {unique_count:,}")
            print(f"  Cardinality: {card_color}{cardinality}{Colors.ENDC}")
            
            # Type-specific stats
            if col_type in ['integer', 'float', 'numeric_string']:
                stats = safe_numeric_stats(df[col], col_type)
                if stats:
                    print(f"  Range: [{stats['min']:.2f} → {stats['max']:.2f}]")
                    print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                    print(f"  Quartiles: Q1={stats['q25']:.2f}, Median={stats['median']:.2f}, Q3={stats['q75']:.2f}")
                    
                    if stats['outlier_count'] > 0:
                        print(f"  ⚠️  Outliers: {stats['outlier_count']:,} ({stats['outlier_pct']:.1f}%)")
            
            elif col_type == 'datetime':
                try:
                    dt_col = pd.to_datetime(df[col], errors='coerce').dropna()
                    if len(dt_col) > 0:
                        print(f"  Date range: {dt_col.min().strftime('%Y-%m-%d')} → {dt_col.max().strftime('%Y-%m-%d')}")
                        print(f"  Span: {(dt_col.max() - dt_col.min()).days} days")
                        
                        # Detect frequency
                        if len(dt_col) > 1:
                            gaps = dt_col.sort_values().diff().dropna()
                            if len(gaps) > 0:
                                mode_gap = gaps.mode()[0] if len(gaps.mode()) > 0 else gaps.median()
                                
                                if mode_gap.days == 0:
                                    freq = "MULTIPLE PER DAY"
                                elif mode_gap.days <= 1:
                                    freq = "DAILY"
                                elif mode_gap.days <= 7:
                                    freq = "WEEKLY"
                                elif mode_gap.days <= 31:
                                    freq = "MONTHLY"
                                else:
                                    freq = "IRREGULAR"
                                
                                print(f"  Frequency: {freq}")
                except:
                    print("  Could not parse datetime values")
            
            elif col_type == 'categorical':
                try:
                    value_counts = df[col].value_counts()
                    print(f"  Top 3 values:")
                    for val, count in value_counts.head(3).items():
                        pct = (count / len(df)) * 100
                        # Truncate long values
                        val_str = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                        print(f"    • '{val_str}': {count:,} ({pct:.1f}%)")
                except:
                    print("  Could not compute value counts")
            
            elif col_type == 'text':
                try:
                    text_lengths = df[col].dropna().astype(str).str.len()
                    if len(text_lengths) > 0:
                        print(f"  Text length: min={text_lengths.min()}, avg={text_lengths.mean():.1f}, max={text_lengths.max()}")
                        
                        # Check for patterns
                        sample = df[col].dropna().head(100).astype(str)
                        
                        # Email check
                        if sample.str.contains(r'@[a-zA-Z]').sum() > len(sample) * 0.5:
                            print(f"  📧 Likely contains: EMAIL addresses")
                        # URL check
                        elif sample.str.contains(r'https?://').sum() > len(sample) * 0.5:
                            print(f"  🔗 Likely contains: URLs")
                        # Phone check
                        elif sample.str.match(r'^[\d\s\-\(\)]+$').sum() > len(sample) * 0.5:
                            print(f"  📞 Likely contains: Phone numbers")
                except:
                    print("  Could not analyze text patterns")
            
            elif col_type == 'boolean':
                try:
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        value_counts = non_null.value_counts()
                        print(f"  Distribution:")
                        for val, count in value_counts.items():
                            pct = (count / len(non_null)) * 100
                            print(f"    • {val}: {count:,} ({pct:.1f}%)")
                except:
                    print("  Could not compute distribution")
            
            # Store profile
            profiles.append({
                'name': col,
                'type': col_type,
                'nulls': int(null_count),
                'null_pct': null_pct,
                'unique': int(unique_count),
                'cardinality': cardinality.lower()
            })
            
        except Exception as e:
            print_colored(f"  ⚠️  Error profiling column: {str(e)[:100]}", Colors.RED)
            profiles.append({
                'name': col,
                'type': 'unknown',
                'nulls': 0,
                'null_pct': 0,
                'unique': 0,
                'cardinality': 'unknown'
            })
    
    # Correlations for numeric columns
    try:
        numeric_cols = [col for col, prof in zip(df.columns, profiles) 
                       if prof['type'] in ['integer', 'float', 'numeric_string']]
        
        if len(numeric_cols) > 1:
            print_colored("\n📊 Correlations (Numeric Columns):", Colors.CYAN)
            print_separator()
            
            # Convert numeric_string columns
            corr_df = pd.DataFrame()
            for col in numeric_cols:
                if profiles[df.columns.get_loc(col)]['type'] == 'numeric_string':
                    corr_df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''), 
                        errors='coerce'
                    )
                else:
                    corr_df[col] = df[col]
            
            if not corr_df.empty:
                corr_matrix = corr_df.corr()
                
                # Find strong correlations
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7 and not np.isnan(corr_value):
                            strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
                
                if strong_corrs:
                    print("  Strong correlations (|r| > 0.7):")
                    for col1, col2, corr in strong_corrs:
                        symbol = "↗" if corr > 0 else "↘"
                        print(f"    {symbol} {col1} ↔ {col2}: {corr:.3f}")
                else:
                    print("  No strong correlations found")
    except Exception as e:
        print_colored(f"\n⚠️  Could not compute correlations: {str(e)[:100]}", Colors.YELLOW)
    
    # Sample data
    try:
        print_colored("\n📋 Sample Data (First 3 Rows):", Colors.CYAN)
        print_separator()
        # Truncate long columns for display
        display_df = df.head(3).copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str).str[:50]
        print(display_df.to_string())
    except Exception as e:
        print_colored(f"Could not display sample data: {str(e)}", Colors.YELLOW)
    
    return profiles

def generate_llm_summary(df, profiles):
    """Generate a summary suitable for LLM processing"""
    print_colored("\n🤖 LLM-READY SUMMARY", Colors.HEADER)
    print_separator()
    
    summary = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": [],
        "sample_rows": []
    }
    
    # Get sample rows (handle potential issues)
    try:
        sample_df = df.head(3).fillna('')
        # Convert to dict, handling any serialization issues
        for _, row in sample_df.iterrows():
            row_dict = {}
            for col in df.columns:
                try:
                    val = row[col]
                    # Convert to JSON-serializable format
                    if pd.isna(val):
                        row_dict[col] = None
                    elif isinstance(val, (int, float, str, bool)):
                        row_dict[col] = val
                    else:
                        row_dict[col] = str(val)
                except:
                    row_dict[col] = ""
            summary["sample_rows"].append(row_dict)
    except:
        print_colored("  ⚠️  Could not serialize sample rows", Colors.YELLOW)
    
    for col, prof in zip(df.columns, profiles):
        col_summary = {
            "name": col,
            "type": prof['type'],
            "cardinality": prof['cardinality'],
            "null_percentage": round(prof['null_pct'], 2)
        }
        
        # Add type-specific info
        try:
            if prof['type'] in ['integer', 'float']:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_summary["range"] = [float(non_null.min()), float(non_null.max())]
            
            elif prof['type'] == 'categorical':
                top_values = df[col].value_counts().head(3)
                col_summary["top_values"] = [str(v) for v in top_values.index]
        except:
            pass
        
        summary["columns"].append(col_summary)
    
    # Print as formatted JSON
    try:
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print_colored(f"Could not print JSON: {str(e)}", Colors.YELLOW)
        # Try without sample rows
        summary["sample_rows"] = []
        print(json.dumps(summary, indent=2))
    
    # Save to file
    output_file = "data/profile_summary.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print_colored(f"\n✓ Summary saved to: {output_file}", Colors.GREEN)
    except Exception as e:
        print_colored(f"\n⚠️  Could not save summary: {str(e)}", Colors.YELLOW)
    
    return summary

def main():
    """Main function"""
    print_colored("\n" + "="*80, Colors.BOLD)
    print_colored("                CSV PROFILER v1.0 (Robust Edition)", Colors.BOLD)
    print_colored("="*80 + "\n", Colors.BOLD)
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print_colored("Usage: python profiler.py <csv_file>", Colors.YELLOW)
        print("\nExample: python profiler.py sales_data.csv")
        print("\nThis profiler handles:")
        print("  • Encoding issues (UTF-8, Latin-1, etc.)")
        print("  • Inconsistent column counts")
        print("  • Missing values and bad data")
        print("  • Large files (chunked reading)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if file exists
    import os
    if not os.path.exists(csv_file):
        print_colored(f"ERROR: File '{csv_file}' not found!", Colors.RED)
        sys.exit(1)
    
    print(f"Processing: {Colors.BOLD}{csv_file}{Colors.ENDC}")
    
    # Step 1: Read CSV
    df = read_csv_smart(csv_file)
    if df is None:
        print_colored("\nFailed to read CSV. The file may be corrupted or in an unsupported format.", Colors.RED)
        sys.exit(1)
    
    # Step 2: Profile dataset
    profiles = profile_dataset(df)
    
    # Step 3: Generate LLM summary
    summary = generate_llm_summary(df, profiles)
    
    print_colored("\n" + "="*80, Colors.BOLD)
    print_colored("                    ✓ PROFILING COMPLETE!", Colors.GREEN)
    print_colored("="*80, Colors.BOLD)
    
    print("\n📁 Output files generated:")
    print("  • profile_summary.json - Ready for LLM processing")
    print("\n💡 Next step: Send profile_summary.json to your LLM for domain detection")

if __name__ == "__main__":
    main()