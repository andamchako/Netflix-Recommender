# Helper functions 
from config  import *
from resources.helper_functions import *

# Streamlit dependencies
import streamlit as st
import joblib,os

# general
import numpy as np 
import pandas as pd
#import pickle
#import dill as pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from recommenders.recommender import preprocess_description
from recommenders.recommender import recommendations

titles = pd.read_csv('netflix_titles.csv')

def main():
    """Movie Recommender App with Streamlit """
    st.image("resources/images/_header.jpeg", width=700)
    st.title("NetFliks Movie Recommender")
    st.write('-----------------------------------------------')
        
    # choose movie from list
    st.write('### Enter Your Favorite Movie')
    movie = st.text_input('Movie Choice:')

    if st.button("Recommend"):
        try:
            with st.spinner('Crunching the numbers...'):
                #movie = preprocess_description(titles['descrition'])
                top_recommendations = recommendations(movie)
            
            st.title("We think you'll like:")
            for i,j in enumerate(top_recommendations):
                st.subheader(str(i+1)+'. '+j)
        except:
            st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

if __name__ == '__main__':
    main()                             