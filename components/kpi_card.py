import streamlit as st

def kpi_card(title: str, value, description: str = ""):
    with st.container():
        st.metric(label=title, value=value, delta=None)
        if description:
            st.caption(description)
