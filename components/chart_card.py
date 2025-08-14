import streamlit as st
import pandas as pd

def render_chart(chart_config: dict, df: pd.DataFrame):
    """
    chart_config example:
    {
      "name": "Revenue Over Time",
      "type": "line",
      "x": "invoicedate",
      "y": "SUM(quantity * unitprice)",
      "description": "Track revenue trends across time"
    }
    """
    st.markdown(f"#### 📊 {chart_config['name']}")
    chart_type = chart_config["type"].lower()

    if chart_type == "line":
        st.line_chart(df.set_index(chart_config["x"])[chart_config["y"]])
    elif chart_type == "bar":
        st.bar_chart(df.set_index(chart_config["x"])[chart_config["y"]])
    elif chart_type == "histogram":
        st.bar_chart(df[chart_config["x"]])
    else:
        st.warning(f"Chart type '{chart_type}' not supported yet.")

    if "description" in chart_config:
        st.caption(chart_config["description"])
