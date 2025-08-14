import streamlit as st

def render_header(domain: str):
    with st.container():
        col1, col2 = st.columns([3, 1])  # wide left, narrow right
        with col1:
            st.markdown(f"### 🌐 {domain.title()} Dashboard")
        with col2:
            st.text_input("Search / Notes", placeholder="Type something...")
