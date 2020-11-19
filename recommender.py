import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processor import preprocess_description

movies = pd.read_csv('movies.csv')

import re
from string import punctuation
import nltk
nltk.download(['averaged_perceptron_tagger','punkt','wordnet','stopwords'])
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 
stop_words = set(stopwords.words('english')) 
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

def preprocess_description(text): 
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
          
    description = ' '.join(temp_sent) 
    description = description.replace("n't", " not") 
    description = description.replace("'m", " am") 
    description = description.replace("'s", " is") 
    description = description.replace("'re", " are") 
    description = description.replace("'ll", " will") 
    description = description.replace("'ve", " have") 
    description = description.replace("'d", " would") 
    return description

def recommendations(title, cosine_sim): 
    movies["plot"]= movies["description"].apply(preprocess_description)

    # Vectorizing pre-processed movie plots using TF-IDF 
    tfidfvec = TfidfVectorizer() 
    tfidf_movieid = tfidfvec.fit_transform((movies["plot"])) 
  
    # Finding cosine similarity between vectors 
    cosine_sim = cosine_similarity(tfidf_movieid, tfidf_movieid)

    # Storing indices of the data 
    indices = pd.Series(movies.index) 
    
    recommended_movies = [] 
    index = indices[indices == title].index[0] 
    similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending = False) 
    top_10_movies = list(similarity_scores.index) 
    for i in top_10_movies: 
        recommended_movies.append(list(movies.index)[i]) 
    #recommended_movies=list(movies['title'].iloc[indices])
    return recommended_movies