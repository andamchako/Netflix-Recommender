"""
    Streamlit web developer application to recommend movies.
    Author: Anda Mchako.
"""
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
#from utils.text_processor import preprocess_description

#titles = pd.read_csv('netflix_titles.csv')

#import re
#from string import punctuation
#import nltk
#nltk.download(['averaged_perceptron_tagger','punkt','wordnet','stopwords'])
#from nltk.corpus import stopwords 
#from nltk.stem import WordNetLemmatizer 

##lemmatizer = WordNetLemmatizer() 
#stop_words = set(stopwords.words('english')) 
#VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

# custom functions
#from recommenders.recommender import preprocess_description
from recommenders.recommender import recommendations

#load data
titles = pd.read_csv('netflix_titles.csv')

#unpickle model
#model_load_path = "model.pkl"
#with open(model_load_path,'rb') as file:
 #   recommendations = pickle.load(file)

# team slides page
team_page   = read_file("profile.html")

# The main function where we will build the actual app
def main():
    """Movie Recommender App with Streamlit """
    st.image("resources/images/_header.jpeg", width=700)
    st.title("NetFliks Movie Recommender")
    st.write('-----------------------------------------------')

    page_options = ["Recommender System","Solution Overview", "About The Author"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    
    if page_selection == "Recommender System":
        
        # choose movie from list
        st.write('### Enter Your Favorite Movie')
        movie = st.selectbox('Movie Choice',titles['title'])

        if st.button("Recommend"):
            try:
                with st.spinner('Crunching the numbers...'):
                    #movie = preprocess_description
                    top_recommendations = recommendations(movie)
            
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    if page_selection == "Solution Overview":
        #st.title("#Solution Overview")
        st.image("resources/images/0.jpg", width=200)
        st.write("For the recommendations made in this app I used a movie similarity content based filtering.\
                The following is an overview of how it works.")
        
        st.write("Here I made recommendations based on how similar the plot description\
                of a movie is to other movies. I used Python's nltk librabry to remove all the stopwords\
                in the target text thus leaving only keywords. I then used Scikit-Learn's TF-IDF vectorizer\
                and used the vectorized form of the descriptions to find the cosine similarity between\
                descriptions.\
                The next steps for this algorithm is to improve it so that it takes a lot\
                more into consideration than just the plot descriptions, I'd like it to consider cast members,\
                directors and genres. ")
        st.write("For more on recommener systems visit:\
            https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/")

    if page_selection == "About The Author":
        #st.write("## Meet Anda")
        #st.image("resources/images/Picture1.jpg", width=200)
        st.markdown(team_page, unsafe_allow_html=True)
        
        local_css('styles.css')

if __name__ == '__main__':
    main()