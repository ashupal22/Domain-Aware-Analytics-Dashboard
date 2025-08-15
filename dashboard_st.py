# app.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from components.utils import eval_formula
from streamlit.runtime.scriptrunner import RerunException, get_script_run_ctx
from datetime import datetime
import numpy as np

# Configure page
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding: 1rem 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .kpi-description {
        font-size: 0.85rem;
        color: #95a5a6;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .filter-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .filter-header::before {
        content: "🔍";
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .section-header::before {
        content: "";
        width: 4px;
        height: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-right: 1rem;
        border-radius: 2px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric Styling Override */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e1e8ed;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Data Table Styling */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Search Box Styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e1e8ed;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to safely rerun the app
def rerun_app():
    try:
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Rerun failed: {e}")

# ---------- Load data ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/data.csv", sep=None, engine="python")
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def load_plan():
    try:
        with open("data/dashboard_plan.json", "r") as f:
            plan = json.load(f)
        return plan
    except Exception as e:
        st.error(f"❌ Error loading plan: {e}")
        return {"domain": "Dashboard", "kpis": [], "charts": []}

# ---------- Data Processing Functions ----------
def format_number(value):
    """Format numbers for display"""
    if isinstance(value, (int, float)):
        if abs(value) >= 1e9:
            return f"{value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.2f}"
    return str(value)

def create_enhanced_chart(chart, x, y, chart_type):
    """Create enhanced charts with better styling"""
    fig = None
    
    # Common styling
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    if chart_type == "line":
        fig = px.line(x=x, y=y, title=chart["name"])
        fig.update_traces(line=dict(width=3, color=colors[0]))
        
    elif chart_type == "bar":
        fig = px.bar(x=x, y=y, title=chart["name"])
        fig.update_traces(marker_color=colors[0])
        
    elif chart_type == "histogram":
        fig = px.histogram(x=x, title=chart["name"], nbins=30)
        fig.update_traces(marker_color=colors[0], opacity=0.8)
        
    elif chart_type == "pie":
        fig = px.pie(values=y, names=x, title=chart["name"])
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(colors=colors, line=dict(color='white', width=2))
        )
    
    if fig:
        # Apply consistent styling
        fig.update_layout(
            title_font_size=16,
            title_font_color='#2c3e50',
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=True if chart_type == "pie" else False
        )
        
        # Update axes styling
        if chart_type not in ["pie"]:
            fig.update_xaxes(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                linecolor='rgba(0,0,0,0.2)'
            )
            fig.update_yaxes(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1,
                linecolor='rgba(0,0,0,0.2)'
            )
    
    return fig

# Load data and plan
with st.spinner('🔄 Loading dashboard data...'):
    df = load_data()
    plan = load_plan()

# ---------- Dashboard Header ----------
st.markdown(f"""
<div class="dashboard-header">
    <h3 class="dashboard-title">{plan.get("domain", "Analytics Dashboard").title()}</h3>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.markdown('<div class="filter-header">Filters & Controls</div>', unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        rerun_app()
    
    st.markdown("---")
    
    # Initialize filters
    filtered_df = df.copy()
    applied_filters = {}
    
    # Determine filterable columns
    filterable_categorical = []
    filterable_numeric = []

    for f in plan['filters'] :
        if f['type'] == 'categorical' :
            filterable_categorical.append(f["column"])

    
    # Categorical filters
    if filterable_categorical:
        st.markdown("### 📂 Categories")
        for col in filterable_categorical:
            options = df[col].dropna().unique().tolist()
            selected = st.multiselect(
                f"{col.title()}",
                options,
                default=options,
                key=f"cat_{col}"
            )
            applied_filters[col] = selected
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
    
    # Numeric sliders
    if filterable_numeric:
        st.markdown("### 📊 Ranges")
        for col in filterable_numeric:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected_range = st.slider(
                f"{col.title()}",
                min_val,
                max_val,
                (min_val, max_val),
                key=f"num_{col}"
            )
            applied_filters[col] = selected_range
            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) & 
                (filtered_df[col] <= selected_range[1])
            ]
    
    # Data summary
    st.markdown("---")
    st.markdown("### 📈 Data Summary")
    total_records = len(df)
    filtered_records = len(filtered_df)
    st.metric("Total Records", f"{total_records:,}")
    st.metric("Filtered Records", f"{filtered_records:,}")
    if total_records > 0:
        filter_percentage = (filtered_records / total_records) * 100
        st.progress(filter_percentage / 100)
        st.caption(f"{filter_percentage:.1f}% of data shown")

# ---------- Search and Quick Actions ----------
col1, col2, col3 = st.columns([3, 2, 1])
with col3:
    search_query = st.text_input("🔍 Search dashboard", placeholder="Search metrics, charts, or data...")
# with col2:
#     view_mode = st.selectbox("👁️ View Mode", ["Standard", "Compact", "Detailed"], index=0)
with col1:
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)

if auto_refresh:
    st.rerun()

# ---------- KPI Section ----------
# st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

kpi_defs = plan.get("kpis", [])
if kpi_defs:
    # Create KPI columns
    kpi_cols = st.columns(min(len(kpi_defs), 4))
    
    for i, kpi in enumerate(kpi_defs[:4]):
        with kpi_cols[i]:
            try:
                value = eval_formula(filtered_df, kpi["formula"], as_scalar=True)
                formatted_value = format_number(value)
                
                # Calculate trend (mock data for demonstration)
                trend_value = np.random.uniform(-5, 15)
                delta_color = "normal" if trend_value >= 0 else "inverse"
                
                st.metric(
                    label=kpi["name"],
                    value=formatted_value,
                    delta=f"{trend_value:+.1f}%",
                    delta_color=delta_color,
                    help=kpi.get("description", "")
                )
            except Exception as e:
                st.error(f"Error calculating {kpi['name']}: {e}")
else:
    st.info("📊 No KPIs configured. Add KPI definitions to your dashboard plan.")

# ---------- Charts Section ----------
st.markdown('<div class="section-header">Data Visualizations</div>', unsafe_allow_html=True)

charts = plan.get("charts", [])
if not charts:
    st.info("📈 No charts configured. Add chart definitions to your dashboard plan.")
else:
    # Helper to calculate slots
    def calculate_slots(chart_type, x):
        if chart_type.lower() == "pie":
            return 1
        n = len(pd.unique(x)) if x is not None and not x.empty else 0
        if n <= 5:
            return 1
        elif n <= 20:
            return 2
        else:
            return 3
    
    # Filter charts based on search
    if search_query:
        charts = [c for c in charts if search_query.lower() in c["name"].lower()]
    
    # Arrange charts dynamically
    i = 0
    while i < len(charts):
        remaining = len(charts) - i
        row_slots = []
        row_charts = []
        used_slots = 0
        j = 0
        
        while j < remaining and used_slots < 3:
            chart = charts[i + j]
            try:
                x = eval_formula(filtered_df, chart.get("x"), as_scalar=False)
                y = eval_formula(filtered_df, chart.get("y"), as_scalar=False) if "y" in chart else None
                chart_type = chart["type"]
                
                slots_needed = calculate_slots(chart_type, x)
                slots_needed = min(slots_needed, 3 - used_slots)
                
                row_slots.append(slots_needed)
                row_charts.append((chart, x, y, chart_type))
                used_slots += slots_needed
                j += 1
            except Exception as e:
                st.error(f"Error processing chart '{chart['name']}': {e}")
                j += 1
        
        if row_charts:
            # Create columns for this row
            row = st.columns(row_slots)
            for k, (chart, x, y, chart_type) in enumerate(row_charts):
                with row[k]:
                    with st.container():
                        # Skip chart if x or y empty
                        if x.empty or (y is not None and y.empty):
                            st.warning(f"⚠️ '{chart['name']}' has no data with current filters.")
                            continue
                        
                        try:
                            fig = create_enhanced_chart(chart, x, y, chart_type)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{k}")
                                if "description" in chart:
                                    st.caption(f"💡 {chart['description']}")
                        except Exception as e:
                            st.error(f"Error rendering chart: {e}")
        
        i += j

# ---------- Data Export Section ----------
# if view_mode == "Detailed":
#     st.markdown('<div class="section-header">Data Export & Details</div>', unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         if st.button("📥 Download CSV", use_container_width=True):
#             csv = filtered_df.to_csv(index=False)
#             st.download_button(
#                 label="💾 Save Filtered Data",
#                 data=csv,
#                 file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv"
#             )
    
#     with col2:
#         if st.button("📋 Copy to Clipboard", use_container_width=True):
#             st.code(filtered_df.to_csv(index=False), language="csv")
    
#     with col3:
#         if st.button("🔍 Show Raw Data", use_container_width=True):
#             st.dataframe(
#                 filtered_df,
#                 use_container_width=True,
#                 height=400
#             )

# ---------- Footer ----------
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>"
        f"Dashboard generated at {datetime.now().strftime('%I:%M %p')} • "
        f"{len(filtered_df):,} records displayed"
        "</div>",
        unsafe_allow_html=True
    )