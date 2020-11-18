# Importing packages and Data
import os
import pandas as pd
import numpy as np

import nltk 
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet') 

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
  
from nltk.corpus import stopwords 
nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'} 

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

os.path.exists('../resources/data/movies.csv')

# Importing data

movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',') # Remove this to be modular
#ratings = pd.read_csv('resources/data/ratings.csv') # Remove this to be modular
movies.fillna('')

# Required columns - Title and movie plot
content = movies[["title", "description"]]           
content = content.set_index('title')

def preprocess_sentences(text): 
    text = text.lower() 
    temp_sent =[] 
    words = nltk.word_tokenize(text) 
    tags = nltk.pos_tag(words) 
    
    for i, word in enumerate(words): 
        if tags[i][1] in VERB_CODES:  
            lemmatized = lemmatizer.lemmatize(word, 'v') 
        else: 
            lemmatized = lemmatizer.lemmatize(word) 
        if lemmatized not in stop_words and lemmatized.isalpha(): 
            temp_sent.append(lemmatized) 
          
    finalsent = ' '.join(temp_sent) 
    finalsent = finalsent.replace("n't", " not") 
    finalsent = finalsent.replace("'m", " am") 
    finalsent = finalsent.replace("'s", " is") 
    finalsent = finalsent.replace("'re", " are") 
    finalsent = finalsent.replace("'ll", " will") 
    finalsent = finalsent.replace("'ve", " have") 
    finalsent = finalsent.replace("'d", " would") 
    return finalsent

def movie_model(list_title):
    # initializing the empty list of recommended movies
    
    content["plot"]= content["description"].apply(preprocess_sentences)
   
    # Vectorizing pre-processed movie plots using TF-IDF 
    tfidfvec = TfidfVectorizer() 
    tfidf_movieid = tfidfvec.fit_transform((content["plot"])) 
  
    # Finding cosine similarity between vectors     
    cos_sim = cosine_similarity(tfidf_movieid, tfidf_movieid)
    indices = pd.Series(content.index)
    # gettin the index of the movie that matches the title
    idx_1 = indices[indices == list_title[0]].index[0]
    idx_2 = indices[indices == list_title[1]].index[0]
    idx_3 = indices[indices == list_title[2]].index[0]
    # creating a Series with the similarity scores in descending order
    rank_1 = cos_sim[idx_1]
    rank_2 = cos_sim[idx_2]
    rank_3 = cos_sim[idx_3]
    # calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)

    #listings.drop_duplicates(inplace=True)

def recommendations(title, cosine_sim = cos_sim): 
    recommended_movies = [] 
    index = list(listings.index) 
    similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending = False) 
    top_10_movies = list(similarity_scores.iloc[1:11].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_10_movies,[idx_1,idx_2,idx_3]) 
    for i in top_10_movies: 
        recommended_movies.append(list(content.index)[i]) 
    return recommended_movies
