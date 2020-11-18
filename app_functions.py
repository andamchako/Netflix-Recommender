"""

    These functions will help us load pages for the app

"""

import streamlit as st 
from pathlib import Path

# Function to load 'Meet the team' page
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

# Function to load CSS to style the 'Meet the team' page
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)