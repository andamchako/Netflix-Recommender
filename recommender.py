"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
from pathlib import Path

# Custom Libraries
from app_functions import *
from utils.data_loader import load_movie_titles
from recommenders.movie_recommender import movies_model
from recommenders.series_recommender import series_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# Load pages
team = read_markdown_file("resources/pages/meet_the_team.html") 
slides = read_markdown_file("resources/pages/slides.html") 

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Data analysis and plots",
                    "Meet the team", "Pitch"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Movie Recommender',
                        'Series Recommender'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list)
        movie_2 = st.selectbox('Second Option',title_list)
        movie_3 = st.selectbox('Third Option',title_list)
        # show_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        # movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        # movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Movie Recommender':
            top_recommendations = movie_model(fav_movies)
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = movie_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Series Recommender':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = series_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("For the recommendations made in this app we used two approaches: \
                user similarity (collaborative based filtering) and movie similarity (content based filtering).\
                The following is an overview of how they work.")
        st.title("Content based")
        st.write("Here we made recommendations based on how similar the properties\
                or features of a movie are to other movies. We used movies with similar\
                themes (in the same genre) or have similar actors to predict what \
                the user will like. If you make your movie selection based on \
                actors you love the recommendations will reflect that, and \
                if you select movies based on a genre you prefer on a \
                specific day the recommendations will reflect that.\
                This algorithm takes a lot more into consideration than just ratings from other users. ")
        st.image("resources/imgs/plots/content_model.jpg", width=450)
        st.title("Collaborative based")
        st.write("This algorithm is based on past interactions between users and the\
                movies in the database. We make recommendations based on movies that those users rated.\
                This method takes into account movies that users with similar tastes liked and makes a \
                recommendation on that basis and no additional information.")
        st.image("resources/imgs/plots/collab_model.jpg", width=650)

    if page_selection == "Data analysis and plots":
        st.title("Data analysis") 
        if st.checkbox("Ratings insights"):
            st.subheader("These plots give insights about the ratings given for the movies")
            st.write("This is a count of movies that have been given a certain rating with 5 being \
                    the highest rating that a movie can get. Most of the movies have been given a \
                    rating of 4 which means that the majority of people enjoy most of the movies in the database \
                    The poorly rated movies have ratings from 0.5-2.5 and they all have a low number of movies. \
                    The reason could be that a poorly rated movie is less likely to be watched by a lot of people.")
            st.image("resources/imgs/plots/ratings_distribution.png", width=650)
            st.write("These are the 20 most rated movies. In the top 10 we only have movies from \
                    the 90s, with 1994 taking the top 3 spots.")
            st.image("resources/imgs/plots/highest_rated_movies.png", width=650)
        if st.checkbox("Movie insights"):
            st.subheader("A number of factors influence movie choices and below we take a look at \
                    some of those factors such as popular themes, actors, directors and era")
            st.write("The average movie runtime is 116.1 minutes which equates to ~1.9 hours.")
            st.image("resources/imgs/plots/runtime.png", width=650)
            st.write("Drama holds the most number of movies in the database followed by comedy and action.")
            st.image("resources/imgs/plots/number_of_movies_by_genre2.png", width=600)
            st.write("The graph below shows the distribution on movies in the dataset. At first glance, \
                it is clear that the 2010s have the highest number of movies released in one decade.")
            st.image("resources/imgs/plots/movies_per_era.png", width=650)
            st.write("These are the most popular themes. The keywords are a reflection of the top 3 genres \
                    in the database (drama, comedy and action). If you watch movies in these genres it is \
                    likely that the movie will have these keywords and that is why these movies have high age \
                    restictions. The keywords also show what themes people enjoy watching.")
            st.image("resources/imgs/plots/wordcloud2.png", width=650)
            st.image("resources/imgs/plots/director_movies.png", width=650)
            st.write("The graph above shows the number of times movies with specific actors in the dataset \
                    have been rated. Tom Hanks takes the lead with more than 195000 movie ratings to his name.\
                    In second place is Samuel L.Jackson followed by Morgan Freeman in third place. It makes sense \
                    that the top 3 actors with the most ratings associated with their names also star in the top \
                    3 most rated movies (refer to 'most rated movies' section). It is important\
                    to note that most of the movies in this database are American based and therefore \
                    the most popular actors are American.")
            st.image("resources/imgs/plots/frequent_actors.png", width=650)


    if page_selection == "Pitch":
        st.title("Pitch slide deck")
        st.markdown(slides, unsafe_allow_html=True)

    if page_selection == "Meet the team":
        st.title("Meet the data science team")
        st.markdown(team, unsafe_allow_html=True)
        local_css('resources/pages/html_style.css')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

if __name__ == '__main__':
    main()
