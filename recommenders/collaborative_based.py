"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.
    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Import essential packages
import pandas as pd
import numpy as np

# Import surprise packages
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

# Import models
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV

# Import utilities
import os
import pickle
import scipy as sp
import operator

# Data dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing train and test datasets
ratings_df = pd.read_csv('resources/data/ratings.csv').drop(columns=['timestamp'],axis=0)
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')

def load_movie_titles(path_to_movies):
    """Load movie titles from database records.
    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.
    Returns
    -------
    list[str]
        Unordered list of movie titles.
    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_list = df['title'].to_list()
    return movie_list

title_list = load_movie_titles('resources/data/movies.csv')
x = title_list[14930:15200]
y = title_list[25055:25255]
w = title_list[21100:21200]
z = w + x + y
edsa_movies = movies_df[movies_df['title'].isin(z)]

edsa_movies[300:350]

def create_subset(movies_sample,all_movies,all_ratings):
    ratings_ids = all_ratings[all_ratings['movieId'].isin(movies_sample['movieId'].tolist())]
    ratings_subset = all_ratings[all_ratings['userId'].isin(ratings_ids['userId'].tolist())]
    return ratings_subset

ratings_subset = create_subset(edsa_movies,movies_df,ratings_df)

ratings_subset = ratings_subset[:1000000]

model=pickle.load(open('resources/models/SVD1000000.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.
    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.
    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.
    """
    # Data preprosessing
    reader = Reader(rating_scale=(1, 5))
    load_df = Dataset.load_from_df(ratings_subset,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.
    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.
    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.
    """
    # Store the id of users
    id_store=[]
    movie_ids = []
    for movie in movie_list:
        movie_ids.append(int(edsa_movies['movieId'][edsa_movies['title']==movie]))
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_ids:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:100]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

def collab_model(movie_list,top_n):
    """Short summary.
    Parameters
    ----------
    movie_list : type
        Description of parameter `movie_list`.
    top_n : type
        Description of parameter `top_n`.
    Returns
    -------
    type
        Description of returned object.
    """
    # get movie ids for movie_list
    movie_ids = []
    for movie in movie_list:
        movie_ids.append(int(edsa_movies['movieId'][edsa_movies['title']==movie]))

    # Create list of users which would rate these movies highly
    user_ids = pred_movies(movie_list)
    
    # Create dataframe of all the movies that these users have rated
    df_init_users = ratings_subset[ratings_subset['userId'].isin(user_ids)]
    
    # Add new user with ratings to userlist
    new_row1 = {'userId':200000, 'movieId':movie_ids[0], 'rating':5.0}
    new_row2 = {'userId':200000, 'movieId':movie_ids[1], 'rating':5.0}
    new_row3 = {'userId':200000, 'movieId':movie_ids[2], 'rating':4.5}
    df_init_users = df_init_users.append([new_row1,new_row2,new_row3], ignore_index=True)
    
    # Creating Util matrix,replace NANs and transpose
    util_matrix = pd.pivot_table(df_init_users,values='rating',columns='movieId',index='userId')
    util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    util_matrix_norm.fillna(0, inplace=True)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
    
    # Save the utility matrix in scipy's sparse matrix format
    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)
    
    # Compute the similarity matrix using the cosine similarity metric
    user_similarity = cosine_similarity(util_matrix_sparse.T)
    
    # Save the matrix as a dataframe to allow for easier indexing  
    user_sim_df = pd.DataFrame(user_similarity,
                               index = util_matrix_norm.columns,
                               columns = util_matrix_norm.columns)
    
    user = 200000
    k=20
    # Gather the k users which are most similar to the reference user 
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:k+1]
    favorite_user_items = [] # <-- List of highest rated items gathered from the k users  
    most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users

    for i in sim_users:
        # Maximum rating given by the current user to an item 
        max_score = util_matrix_norm.loc[:, i].max()
        # Save the names of items maximally rated by the current user   
        favorite_user_items.append(util_matrix_norm[util_matrix_norm.loc[:, i]==max_score].index.tolist())

    # Loop over each user's favorite items and tally which ones are 
    # most popular overall.
    for item_collection in range(len(favorite_user_items)):
        for item in favorite_user_items[item_collection]:
            if item in most_common_favorites:
                most_common_favorites[item] += 1
            else:
                most_common_favorites[item] = 1
    # Sort the overall most popular items and return the top-N instances
    sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:top_n+1]
    top_N = [x[0] for x in sorted_list]
    
    # Return Movie Names
    recommendations = []
    for movieid in top_N:
        recommendations.append(movies_df[movies_df['movieId']==movieid]['title'].tolist())
    recommendations = [item for sublist in recommendations for item in sublist]
    recommendations = [x for x in recommendations if x not in movie_list]
    recommendations[:top_n]
    return recommendations

