"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv')
movies.dropna(inplace=True)

# Select subset of movies

movies=movies.iloc[0:25257,:]

# Generate dataframe for recommender
movies = pd.merge(movies,ratings,on='movieId',how='left')
movies = pd.merge(movies,imdb,on='movieId',how='left')

# Remove NaN values

movies['rating']=movies['rating'].fillna(0)
movies['title_cast']=movies['title_cast'].fillna('')
movies['genres']=movies['genres'].fillna('')

movies['content'] = movies['genres']+'|'+movies['title_cast']
movies.drop(['userId','timestamp','director','runtime','budget','genres','title_cast','plot_keywords'],axis=1,inplace=True)


def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    
    movies['content'] = movies['content'].str.replace('|', ' ')
    
    # Subset of the data
    # Group data by title to get unique movie titles

    genres = pd.DataFrame(movies.groupby(['title'])['content'].unique())
    rating=pd.DataFrame(movies.groupby(['title'])['rating'].count())
    movies_subset=pd.merge(genres,rating,on='title').sort_values('rating',ascending=False)

    # Function to convert genres column from list to strings

    def string(input):
      string=''
      for i in input:
        string=string+str (i)
      return string

    movies_subset['content']=movies_subset['content'].apply(string) # apply string function

    
    movies_subset = movies_subset[:subset_size] # return subset of dataframe
    
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=12):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    # Use a subset of movies for feature 
    
    movies_subset= data_preprocessing(-1)

    feature=movies_subset['content']

    # Set up vectorizer

    vec=TfidfVectorizer(stop_words='english') # term frequency inverse document frequency

    feature_vec=vec.fit_transform(feature) # fit transform vec into sparse matrix

    # Use cosine similarity to get similarity matrix
    
    sim_matrix =cosine_similarity(feature_vec,feature_vec)


    movies_subset=movies_subset.reset_index() # reset index

    index_map = pd.Series(movies_subset.index,index = movies_subset['title']) # contains a series of the movie titles

    # use for loop for to iterate through movie list

    recommendations_indx=[]

    for i in movie_list:
      
      index = index_map[i] # returns the index of the argument title in index_map
      
      similarity = list(enumerate(sim_matrix[index])) # returns a list of with index and similarity
      
      similarity = sorted(similarity, key=lambda x: x[1], reverse=True) # sorts list in descending order
      
      similarity = similarity[1:top_n] # top 10 recommendations
      
      #return movie names using the mapping series
       
      movie_indices = [i[0] for i in similarity] # obtain indices for recommended movies
       
      recommendations_indx.append(movie_indices)

    #return movie names using the mapping series

    movie_indices = recommendations_indx[0][0:3]+recommendations_indx[1][0:3]+recommendations_indx[2][0:4] # obtain indices for recommended movies

    recommended_movies=list(movies_subset['title'].iloc[movie_indices]) # filter movie_df for titles
    
    return recommended_movies
