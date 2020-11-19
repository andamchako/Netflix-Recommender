import streamlit as st
import joblib,os
from pathlib import Path
from config  import PAGES_PATH

########################################
#           PAGES FUNCTIONS            #    
########################################
def read_file(markdown_file, PAGES_PATH=PAGES_PATH):
    return Path(PAGES_PATH+markdown_file).read_text()

def local_css(file_name):
    with open(PAGES_PATH+file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


# local_css('styles.css')