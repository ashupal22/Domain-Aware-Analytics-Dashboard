"""
Classic Streamlit dashboard with clean design - FIXED CHART NAMES
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Any, Optional
from io import BytesIO
import warnings
import re
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')
from config import ConfigManager
from data_loader import CSVDataLoader

# Page Configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ClassicDashboard:
    """Classic Streamlit dashboard with clean, simple design"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.data_loader = CSVDataLoader()
        self.dashboard_plan = None
        self.df = None
        self.filtered_df = None
        self.profile_data = None
        
        # Initialize session state for search
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        if 'chart_search' not in st.session_state:
            st.session_state.chart_search = ""
        
    def load_dashboard_data(self) -> bool:
        """Load all pipeline outputs"""
        # Load processed data
        if not self._load_processed_data():
            return False
            
        # Load profile data
        if not self._load_profile_data():
            pass  # Silent - not critical
            
        # Load dashboard plan
        if not self._load_dashboard_plan():
            return False
            
        # Create calculated columns before filtering
        self._create_calculated_columns()
        
        self.filtered_df = self.df.copy()
        return True
    
    def _create_calculated_columns(self):
        """Create calculated columns silently"""
        try:
            # Common calculated columns for e-commerce/sales data
            if 'quantity' in self.df.columns and 'unitprice' in self.df.columns:
                self.df['total_sales'] = self.df['quantity'] * self.df['unitprice']
            
            # Check for other common patterns
            if 'price' in self.df.columns and 'qty' in self.df.columns:
                self.df['total_amount'] = self.df['price'] * self.df['qty']
            
            # Revenue calculations
            if 'revenue' not in self.df.columns and 'total_sales' in self.df.columns:
                self.df['revenue'] = self.df['total_sales']
            
            # Extract calculated columns from dashboard plan
            self._extract_calculated_columns_from_plan()
            
        except Exception:
            pass  # Silent failure
    
    def _extract_calculated_columns_from_plan(self):
        """Extract and create calculated columns from dashboard plan"""
        if not self.dashboard_plan:
            return
        
        # Check KPIs for calculated columns
        kpis = self.dashboard_plan.get('kpis', [])
        for kpi in kpis:
            formula = kpi.get('formula', '')
            self._create_columns_from_formula(formula)
        
        # Check charts for calculated columns
        charts = self.dashboard_plan.get('charts', [])
        for chart in charts:
            y_column = chart.get('y_column', '')
            if y_column and '(' in y_column and ')' in y_column:
                self._create_columns_from_formula(y_column)
    
    def _create_columns_from_formula(self, formula: str):
        """Create calculated columns from formula references silently"""
        try:
            # Pattern to match: (df['quantity'] * df['unitprice'])
            pattern = r"\(df\['(\w+)'\]\s*\*\s*df\['(\w+)'\]\)"
            matches = re.findall(pattern, formula)
            
            for col1, col2 in matches:
                if col1 in self.df.columns and col2 in self.df.columns:
                    new_col_name = f"{col1}_{col2}_product"
                    if new_col_name not in self.df.columns:
                        self.df[new_col_name] = self.df[col1] * self.df[col2]
                        
                        # Also create the exact formula reference as column name
                        formula_col = f"(df['{col1}'] * df['{col2}'])"
                        if formula_col not in self.df.columns:
                            self.df[formula_col] = self.df[col1] * self.df[col2]
            
        except Exception:
            pass  # Silent failure
    
    def _load_processed_data(self) -> bool:
        """Load processed CSV data silently"""
        # Try processed data first
        if self.data_loader.load_processed_data():
            self.df = self.data_loader.get_data()
            self.df = self._fix_arrow_compatibility(self.df)
            return True
        
        # Fallback to original file
        if os.path.exists(self.config.data.processed_file):
            success, message = self.data_loader.load_csv(self.config.data.processed_file, save_processed=False)
            if success:
                self.df = self.data_loader.get_data()
                self.df = self._fix_arrow_compatibility(self.df)
                return True
        
        return False
    
    def _load_profile_data(self) -> bool:
        """Load profile JSON data"""
        try:
            if os.path.exists(self.config.data.profile_file):
                with open(self.config.data.profile_file, 'r') as f:
                    self.profile_data = json.load(f)
                return True
        except Exception:
            pass
        return False
    
    def _load_dashboard_plan(self) -> bool:
        """Load dashboard plan JSON silently"""
        try:
            if os.path.exists(self.config.data.dashboard_plan_file):
                with open(self.config.data.dashboard_plan_file, 'r') as f:
                    self.dashboard_plan = json.load(f)
                return True
            else:
                return False
        except Exception:
            return False
    
    def _fix_arrow_compatibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive fix for Arrow table compatibility"""
        try:
            fixed_df = df.copy()
            
            # Step 1: Fix datetime columns
            for col in fixed_df.columns:
                if fixed_df[col].dtype == 'object':
                    # Check for datetime patterns
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                        try:
                            dt_series = pd.to_datetime(fixed_df[col], errors='coerce')
                            fixed_df[col] = dt_series.dt.strftime('%Y-%m-%d %H:%M:%S')
                            fixed_df[col] = fixed_df[col].fillna('')
                            continue
                        except:
                            pass
                    
                    # Clean object columns
                    try:
                        fixed_df[col] = fixed_df[col].astype(str).replace('nan', '').replace('None', '')
                    except:
                        fixed_df[col] = fixed_df[col].apply(lambda x: str(x) if x is not None else '')
            
            # Step 2: Fix numeric columns
            for col in fixed_df.columns:
                if fixed_df[col].dtype == 'object':
                    try:
                        numeric_series = pd.to_numeric(fixed_df[col], errors='coerce')
                        if numeric_series.notna().sum() > len(fixed_df) * 0.5:
                            fixed_df[col] = numeric_series.fillna(0)
                        else:
                            fixed_df[col] = fixed_df[col].astype(str).replace('nan', '')
                    except:
                        fixed_df[col] = fixed_df[col].astype(str).replace('nan', '')
            
            # Step 3: Handle datetime dtypes
            for col in fixed_df.columns:
                if str(fixed_df[col].dtype).startswith('datetime64'):
                    fixed_df[col] = fixed_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    fixed_df[col] = fixed_df[col].fillna('')
                elif fixed_df[col].dtype == 'object':
                    fixed_df[col] = fixed_df[col].astype(str).replace('nan', '').replace('None', '')
            
            # Remove empty columns
            fixed_df = fixed_df.dropna(axis=1, how='all')
            
            return fixed_df
            
        except Exception as e:
            st.error(f"Error fixing Arrow compatibility: {e}")
            return df
    
    def render_header(self):
        """Render simple header"""
        # Get domain name
        domain_name = "Analytics Dashboard"
        if self.dashboard_plan and isinstance(self.dashboard_plan, dict):
            domain_info = self.dashboard_plan.get('domain', {})
            if isinstance(domain_info, dict):
                domain_name = domain_info.get('name', 'Analytics Dashboard')
        
        # Header with search
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title(f"{domain_name}")
        with col2:
            st.session_state.chart_search = st.text_input(
                "Search charts", 
                value=st.session_state.chart_search,
                placeholder="Search charts...",
                key="chart_search_input"
            )
    
    def execute_formula_safely(self, formula: str) -> float:
        """Execute KPI formula safely with improved column mapping"""
        safe_globals = {
            'pd': pd,
            'np': np,
            'df': self.filtered_df,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'mean': np.mean,
        }
        
        try:
            if not formula:
                return 0
            
            # Fix formula with improved column mapping
            fixed_formula = self._fix_formula_columns(formula)
            
            # Execute formula
            result = eval(fixed_formula, safe_globals)
            return float(result) if result is not None else 0
            
        except Exception:
            return 0
    
    def _fix_formula_columns(self, formula: str) -> str:
        """Improved formula column fixing"""
        try:
            actual_columns = list(self.filtered_df.columns)
            
            # If the exact formula column exists (like calculated columns), use it
            if formula in actual_columns:
                return f"df['{formula}']"
            
            # Handle formula patterns like (df['quantity'] * df['unitprice'])
            pattern = r"\(df\['(\w+)'\]\s*\*\s*df\['(\w+)'\]\)"
            matches = re.findall(pattern, formula)
            
            for col1, col2 in matches:
                if col1 in actual_columns and col2 in actual_columns:
                    # Replace with actual calculation
                    old_pattern = f"(df['{col1}'] * df['{col2}'])"
                    new_calculation = f"(df['{col1}'] * df['{col2}'])"
                    formula = formula.replace(old_pattern, new_calculation)
                else:
                    # Try to find similar columns
                    alt_col1 = self._find_similar_column(col1, actual_columns)
                    alt_col2 = self._find_similar_column(col2, actual_columns)
                    
                    if alt_col1 and alt_col2:
                        old_pattern = f"(df['{col1}'] * df['{col2}'])"
                        new_calculation = f"(df['{alt_col1}'] * df['{alt_col2}'])"
                        formula = formula.replace(old_pattern, new_calculation)
            
            # Handle simple column references
            simple_pattern = r"df\['(\w+)'\]"
            matches = re.findall(simple_pattern, formula)
            
            for col in matches:
                if col not in actual_columns:
                    alt_col = self._find_similar_column(col, actual_columns)
                    if alt_col:
                        formula = formula.replace(f"df['{col}']", f"df['{alt_col}']")
            
            return formula
            
        except Exception:
            return formula
    
    def _find_similar_column(self, target_col: str, available_cols: List[str]) -> Optional[str]:
        """Find similar column names"""
        target_lower = target_col.lower()
        
        # Exact match (case insensitive)
        for col in available_cols:
            if col.lower() == target_lower:
                return col
        
        # Contains match
        for col in available_cols:
            if target_lower in col.lower() or col.lower() in target_lower:
                return col
        
        # Common mappings
        mappings = {
            'quantity': ['qty', 'amount', 'count', 'volume'],
            'unitprice': ['price', 'unit_price', 'cost', 'rate'],
            'total': ['amount', 'sum', 'total_amount'],
            'sales': ['revenue', 'amount', 'total'],
            'revenue': ['sales', 'total', 'amount']
        }
        
        if target_lower in mappings:
            for alt in mappings[target_lower]:
                for col in available_cols:
                    if alt in col.lower():
                        return col
        
        return None
    
    def _format_axis_label(self, column_ref: str) -> str:
        """Format axis labels to be human-readable"""
        if not column_ref:
            return ""
        
        # Handle calculated column references like (df['quantity'] * df['unitprice'])
        pattern = r"\(df\['(\w+)'\]\s*\*\s*df\['(\w+)'\]\)"
        match = re.search(pattern, column_ref)
        
        if match:
            col1, col2 = match.groups()
            return f"({col1} × {col2})"
        
        # Handle simple column references
        simple_pattern = r"df\['(\w+)'\]"
        simple_match = re.search(simple_pattern, column_ref)
        
        if simple_match:
            return simple_match.group(1)
        
        return column_ref
    
    def render_kpis(self):
        """Render clean KPI cards"""
        # Safe check for dashboard plan
        if not self.dashboard_plan or not isinstance(self.dashboard_plan, dict):
            return
            
        kpis = self.dashboard_plan.get('kpis', [])
        if not kpis:
            return
        
        # Sort by importance
        sorted_kpis = sorted(kpis, key=lambda x: x.get('importance', 0), reverse=True)
        
        # Simple grid layout
        num_kpis = len(sorted_kpis)
        if num_kpis <= 4:
            cols = st.columns(num_kpis)
        else:
            cols = st.columns(4)
        
        for idx, kpi_config in enumerate(sorted_kpis):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                self._render_kpi_card(kpi_config)
    
    def _render_kpi_card(self, kpi_config: dict):
        """Render simple KPI card using Streamlit metrics"""
        name = kpi_config.get('name', 'KPI')
        Reason = kpi_config.get('reason', '')
        description = kpi_config.get('description', '')
        formula = kpi_config.get('formula', '')
        importance = kpi_config.get('importance', 0)

        try:
            # Execute formula
            value = self.execute_formula_safely(formula)
            formatted_value = self._format_kpi_value(value)
            
            # Create detailed help text
            help_text = f"""
**Description:** {description}

**Reason:** {Reason}

**Formula:** {formula}
            """.strip()
            
            # Use Streamlit's metric widget
            st.metric(
                label=name,
                value=formatted_value,
                help=help_text
            )
            
        except Exception:
            st.metric(
                label=name,
                value="Error",
                help=f"Error calculating KPI: {formula}"
            )
    
    def _format_kpi_value(self, value: float) -> str:
        """Smart KPI value formatting"""
        try:
            abs_value = abs(value)
            if abs_value >= 1_000_000_000:
                return f"{value/1_000_000_000:.1f}B"
            elif abs_value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif abs_value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:,.0f}"
        except:
            return str(value)
    
    def render_charts(self):
        """Render charts in simple grid with search functionality"""
        # Safe check for dashboard plan
        if not self.dashboard_plan or not isinstance(self.dashboard_plan, dict):
            return
            
        charts = self.dashboard_plan.get('charts', [])
        if not charts:
            return
        
        # Filter charts based on search
        filtered_charts = charts
        if st.session_state.chart_search:
            search_term = st.session_state.chart_search.lower()
            filtered_charts = [
                chart for chart in charts 
                if search_term in chart.get('name', '').lower() or 
                search_term in chart.get('description', '').lower() or
                search_term in chart.get('type', '').lower()
            ]
        
        if not filtered_charts:
            st.info(f"No charts found matching '{st.session_state.chart_search}'")
            return
        
        # Sort by importance
        sorted_charts = sorted(filtered_charts, key=lambda x: x.get('importance', 0), reverse=True)
        
        # Render charts with special handling for heatmaps
        i = 0
        while i < len(sorted_charts):
            current_chart = sorted_charts[i]
            
            # If current chart is a heatmap, render it in full width
            if current_chart.get('type') == 'heatmap':
                self._render_chart(current_chart)
                i += 1
            else:
                # For non-heatmap charts, try to pair them
                if i + 1 < len(sorted_charts) and sorted_charts[i + 1].get('type') != 'heatmap':
                    # Two non-heatmap charts per row
                    col1, col2 = st.columns(2)
                    with col1:
                        self._render_chart(sorted_charts[i])
                    with col2:
                        self._render_chart(sorted_charts[i + 1])
                    i += 2
                else:
                    # Single non-heatmap chart in full width (next chart is heatmap or this is the last chart)
                    self._render_chart(sorted_charts[i])
                    i += 1
    
    def _render_chart(self, chart_config: dict):
        """Render individual chart with proper title display"""
        try:
            # Get the chart name for display
            chart_name = chart_config.get('name', 'Chart')
            chart_type = chart_config.get('type', 'bar')
            x_column = chart_config.get('x_column', '')
            y_column = chart_config.get('y_column', '')
            importance = chart_config.get('importance', 0)
            
            # Chart container with title
            with st.container():
                # Display chart name as header
                st.subheader(chart_name)
                
                # Fix column references
                if x_column and x_column not in self.filtered_df.columns:
                    alt_x = self._find_similar_column(x_column, list(self.filtered_df.columns))
                    if alt_x:
                        x_column = alt_x
                    else:
                        st.warning(f"Column '{x_column}' not found")
                        return
                
                if y_column and y_column not in self.filtered_df.columns:
                    if '(' in y_column and ')' in y_column:
                        calc_col = self._handle_calculated_column_reference(y_column)
                        if calc_col:
                            y_column = calc_col
                        else:
                            st.warning(f"Could not resolve column '{y_column}'")
                            return
                    else:
                        alt_y = self._find_similar_column(y_column, list(self.filtered_df.columns))
                        if alt_y:
                            y_column = alt_y
                        else:
                            st.warning(f"Column '{y_column}' not found")
                            return
                
                # Render chart based on type
                if chart_type == 'time_series':
                    self._render_time_series(x_column, y_column, chart_config)
                elif chart_type == 'bar':
                    self._render_bar_chart(x_column, y_column, chart_config)
                elif chart_type == 'histogram':
                    self._render_histogram(x_column, chart_config)
                elif chart_type == 'scatter':
                    self._render_scatter(x_column, y_column, chart_config)
                elif chart_type == 'heatmap':
                    self._render_heatmap(chart_config)
                elif chart_type == 'pie':
                    self._render_pie(x_column, y_column, chart_config)
                elif chart_type == 'funnel':
                    self._render_funnel(x_column, y_column, chart_config)
                else:
                    st.dataframe(self.filtered_df.head(5), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering chart '{chart_config.get('name', 'Unknown')}': {str(e)}")
    
    def _handle_calculated_column_reference(self, column_ref: str) -> Optional[str]:
        """Handle calculated column references"""
        try:
            if column_ref in self.filtered_df.columns:
                return column_ref
            
            pattern = r"df\['(\w+)'\]"
            mentioned_cols = re.findall(pattern, column_ref)
            
            if len(mentioned_cols) >= 2:
                col1, col2 = mentioned_cols[0], mentioned_cols[1]
                actual_col1 = self._find_similar_column(col1, list(self.filtered_df.columns))
                actual_col2 = self._find_similar_column(col2, list(self.filtered_df.columns))
                
                if actual_col1 and actual_col2:
                    calc_col_name = f"{actual_col1}_x_{actual_col2}"
                    if calc_col_name not in self.filtered_df.columns:
                        self.filtered_df[calc_col_name] = self.filtered_df[actual_col1] * self.filtered_df[actual_col2]
                        self.df[calc_col_name] = self.df[actual_col1] * self.df[actual_col2]
                    return calc_col_name
            
            return None
        except Exception:
            return None
    
    def _render_time_series(self, x_column: str, y_column: str, chart_config: dict):
        """Simple time series chart"""
        try:
            df_plot = self.filtered_df.copy()
            if df_plot[x_column].dtype == 'object':
                df_plot[x_column] = pd.to_datetime(df_plot[x_column], errors='coerce')
            
            df_agg = df_plot.groupby(x_column)[y_column].sum().reset_index()
            
            # Format axis labels
            x_label = self._format_axis_label(x_column)
            y_label = self._format_axis_label(chart_config.get('y_column', y_column))
            
            # Don't repeat the chart name in the title
            fig = px.line(df_agg, x=x_column, y=y_column)
            fig.update_xaxes(title=x_label)
            fig.update_yaxes(title=y_label)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating time series: {e}")
    
    def _render_bar_chart(self, x_column: str, y_column: str, chart_config: dict):
        """Simple bar chart"""
        try:
            # Format axis labels
            x_label = self._format_axis_label(x_column)
            y_label = self._format_axis_label(chart_config.get('y_column', y_column)) if y_column else "Count"
            
            if not y_column:
                value_counts = self.filtered_df[x_column].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values)
                fig.update_xaxes(title=x_label)
                fig.update_yaxes(title="Count")
            else:
                df_agg = self.filtered_df.groupby(x_column)[y_column].sum().reset_index()
                df_agg = df_agg.nlargest(10, y_column)
                fig = px.bar(df_agg, x=x_column, y=y_column)
                fig.update_xaxes(title=x_label)
                fig.update_yaxes(title=y_label)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating bar chart: {e}")
    
    def _render_histogram(self, x_column: str, chart_config: dict):
        """Simple histogram"""
        try:
            x_label = self._format_axis_label(x_column)
            fig = px.histogram(self.filtered_df, x=x_column, nbins=20)
            fig.update_xaxes(title=x_label)
            fig.update_yaxes(title="Count")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating histogram: {e}")
    
    def _render_scatter(self, x_column: str, y_column: str, chart_config: dict):
        """Simple scatter plot"""
        try:
            x_label = self._format_axis_label(x_column)
            y_label = self._format_axis_label(chart_config.get('y_column', y_column))
            
            fig = px.scatter(self.filtered_df, x=x_column, y=y_column)
            fig.update_xaxes(title=x_label)
            fig.update_yaxes(title=y_label)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")
    
    def _render_heatmap(self, chart_config: dict):
        """Simple heatmap"""
        try:
            numeric_df = self.filtered_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                st.info("Need 2+ numeric columns for heatmap")
                return
            
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, aspect="auto")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
    
    def _render_pie(self, x_column: str, y_column: str, chart_config: dict):
        """Simple pie chart"""
        try:
            x_label = self._format_axis_label(x_column)
            y_label = self._format_axis_label(chart_config.get('y_column', y_column)) if y_column else "Count"
            
            if not y_column:
                value_counts = self.filtered_df[x_column].value_counts().head(8)
                fig = px.pie(values=value_counts.values, names=value_counts.index)
            else:
                df_agg = self.filtered_df.groupby(x_column)[y_column].sum().head(8).reset_index()
                fig = px.pie(df_agg, values=y_column, names=x_column)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating pie chart: {e}")
    
    def _render_funnel(self, x_column: str, y_column: str, chart_config: dict):
        """Simple funnel chart"""
        try:
            x_label = self._format_axis_label(x_column)
            y_label = self._format_axis_label(chart_config.get('y_column', y_column))
            
            df_agg = self.filtered_df.groupby(x_column)[y_column].sum().reset_index()
            df_agg = df_agg.sort_values(y_column, ascending=False)
            
            fig = go.Figure(go.Funnel(
                y=df_agg[x_column],
                x=df_agg[y_column]
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating funnel chart: {e}")
    
    def render_filters(self):
        """Simple sidebar filters"""
        st.sidebar.header("🔍 Filters")
        
        # Safe check for dashboard plan
        if not self.dashboard_plan or not isinstance(self.dashboard_plan, dict):
            st.sidebar.warning("No filters available")
            return
        
        filters = self.dashboard_plan.get('filters', [])
        if not filters:
            st.sidebar.info("No filters configured")
            return
            
        sorted_filters = sorted(filters, key=lambda x: x.get('importance', 0), reverse=True)
        
        for filter_config in sorted_filters:
            try:
                self._render_single_filter(filter_config)
            except Exception:
                pass
        
        # Reset filters button
        if st.sidebar.button("🔄 Reset All Filters"):
            self.filtered_df = self.df.copy()
            st.rerun()
    
    def _render_single_filter(self, filter_config: dict):
        """Render individual filter"""
        filter_name = filter_config.get('name', 'Filter')
        column = filter_config.get('column', '')
        filter_type = filter_config.get('type', 'multiselect')
        
        if column not in self.df.columns:
            alt_column = self._find_similar_column(column, list(self.df.columns))
            if alt_column:
                column = alt_column
            else:
                return
        
        if filter_type == 'date':
            self._render_date_filter(filter_name, column)
        elif filter_type == 'multiselect':
            self._render_multiselect_filter(filter_name, column)
        elif filter_type == 'slider' and pd.api.types.is_numeric_dtype(self.df[column]):
            self._render_slider_filter(filter_name, column)
        else:
            self._render_selectbox_filter(filter_name, column)
    
    def _render_date_filter(self, name: str, column: str):
        """Date filter"""
        try:
            if self.df[column].dtype == 'object':
                date_series = pd.to_datetime(self.df[column], errors='coerce')
            else:
                date_series = self.df[column]
            
            min_date = date_series.min().date()
            max_date = date_series.max().date()
            
            date_range_days = (max_date - min_date).days
            if date_range_days > 90:
                default_start = max_date - timedelta(days=90)
            else:
                default_start = min_date
            
            date_range = st.sidebar.date_input(
                name,
                value=[default_start, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = ((date_series.dt.date >= start_date) & 
                       (date_series.dt.date <= end_date))
                self.filtered_df = self.filtered_df[mask]
        except Exception:
            pass
    
    def _render_multiselect_filter(self, name: str, column: str):
        """Multiselect filter"""
        try:
            unique_values = sorted(self.df[column].dropna().unique())
            
            if len(unique_values) > 100:
                value_counts = self.df[column].value_counts()
                unique_values = value_counts.head(100).index.tolist()
            
            default_selection = unique_values[:min(5, len(unique_values))]
            
            selected = st.sidebar.multiselect(
                name,
                options=unique_values,
                default=default_selection
            )
            
            if selected:
                self.filtered_df = self.filtered_df[self.filtered_df[column].isin(selected)]
        except Exception:
            pass
    
    def _render_slider_filter(self, name: str, column: str):
        """Slider filter"""
        try:
            p1 = float(self.df[column].quantile(0.01))
            p99 = float(self.df[column].quantile(0.99))
            
            values = st.sidebar.slider(
                name,
                min_value=p1,
                max_value=p99,
                value=(p1, p99)
            )
            
            mask = ((self.filtered_df[column] >= values[0]) & 
                   (self.filtered_df[column] <= values[1]))
            self.filtered_df = self.filtered_df[mask]
        except Exception:
            pass
    
    def _render_selectbox_filter(self, name: str, column: str):
        """Selectbox filter"""
        try:
            value_counts = self.df[column].value_counts()
            unique_values = ['All'] + value_counts.head(50).index.tolist()
            
            selected = st.sidebar.selectbox(name, unique_values)
            
            if selected != 'All':
                self.filtered_df = self.filtered_df[self.filtered_df[column] == selected]
        except Exception:
            pass
    
    def render_data_table(self):
        """Render data table section"""
        st.subheader("📋 Data Table")
        
        # Show filtered data info
        st.write(f"Showing {len(self.filtered_df):,} records")
        
        # Display the data
        st.dataframe(
            self.filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = self.filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Main dashboard execution"""
        try:
            # Apply enhanced styling
            st.markdown("""
            <style>
            /* Hide Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            
            /* Improve spacing */
            .main {
                padding-top: 1rem;
            }
            
            /* Style metrics with light colors */
            [data-testid="metric-container"] {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                
                padding: 1.2rem;
                border-radius: 0.75rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.07);
                transition: transform 0.2s ease;
            }
            
            [data-testid="metric-container"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            }
            
            /* Style metric labels */
            [data-testid="metric-container"] [data-testid="metric-container-label"] {
                font-weight: 600;
                color: #495057;
                font-size: 0.9rem;
            }
            
            /* Style metric values */
            [data-testid="metric-container"] [data-testid="metric-container-value"] {
                font-weight: 700;
                color: #212529;
                font-size: 1.8rem;
            }
            
            /* Style charts */
            .js-plotly-plot {
                overflow: hidden;
            }
            
            /* Remove dividers */
            hr {
                display: none;
            }
            
            /* Add spacing between sections */
            .main > div > div > div > div {
                margin-bottom: 2rem;
            }
            
            /* Style search input */
            .stTextInput > div > div > input {
                border-radius: 0.5rem;
                border: 2px solid #e9ecef;
                padding: 0.5rem 1rem;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #0066cc;
                box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
            }
            
            /* Style containers */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Improve title styling */
            h1 {
                color: #212529;
                font-weight: 700;
                margin-bottom: 1.5rem;
            }
            
            /* Style chart titles */
            h3 {
                color: #495057;
                font-weight: 600;
                margin-bottom: 1rem;
                margin-top: 2rem;
            }
            
            /* Add subtle background */
            .main {
                background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
            }
                        
            </style>
            """, unsafe_allow_html=True)
            
            # Load data and dashboard plan
            if not self.load_dashboard_data():
                st.error("❌ Failed to load data. Please run the pipeline first.")
                st.info("Make sure you have:")
                st.write("- Uploaded a CSV file")
                st.write("- Run the data analysis pipeline")
                st.write("- Generated dashboard configuration")
                return
            
            # Render header with search
            self.render_header()
            
            # Add spacing
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            
            # Sidebar filters
            self.render_filters()
            
            # Main content
            if len(self.filtered_df) == 0:
                st.warning("⚠️ No data matches current filters. Please adjust your filter settings.")
                return
            
            # KPIs section
            self.render_kpis()
            
            # Add spacing between sections
            st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
            
            # Charts section
            self.render_charts()
            
            # Add spacing
            st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Dashboard error: {str(e)}")
            
            # Debug information
            with st.expander("🐛 Debug Information"):
                st.write("**Dashboard Status:**")
                st.write(f"- Dashboard plan loaded: {self.dashboard_plan is not None}")
                st.write(f"- Data loaded: {self.df is not None}")
                
                if self.df is not None:
                    st.write(f"- Data shape: {self.df.shape}")
                    st.write(f"- Columns: {list(self.df.columns)}")
                    
                if self.dashboard_plan is not None:
                    st.write("**Dashboard Plan:**")
                    st.json(self.dashboard_plan)
                
                st.write("**Error Details:**")
                st.exception(e)

def main():
    """Main application entry point"""
    try:
        # Initialize and run dashboard
        dashboard = ClassicDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"❌ Application startup error: {str(e)}")
        st.info("Please check:")
        st.write("- Data files are in the correct location")
        st.write("- All required dependencies are installed")
        st.write("- The pipeline has been run successfully")

if __name__ == "__main__":
    main()