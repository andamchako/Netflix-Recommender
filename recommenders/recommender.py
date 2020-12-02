import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#from utils.text_processor import preprocess_description

titles = pd.read_csv('netflix_titles.csv')

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

def recommendations(title): 
    
    # Required columns - Title and movie plot
    content = df[["title", "description"]]           
    content = content.set_index('title')
    
    content["plot"]= content["description"].apply(preprocess_sentences)
    
    # Vectorizing pre-processed movie plots using TF-IDF 
    tfidfvec = TfidfVectorizer() 
    tfidf_movieid = tfidfvec.fit_transform((content["plot"])) 
    
    # Finding cosine similarity between vectors 
    cos_sim = cosine_similarity(tfidf_movieid, tfidf_movieid)
    
    # Storing indices of the data 
    indices = pd.Series(content.index)
    
    recommended_movies = [] 
    index = indices[indices == title].index[0] 
    similarity_scores = pd.Series(cos_sim[index]).sort_values(ascending = False) 
    top_10_movies = list(similarity_scores.iloc[1:11].index) 
    for i in top_10_movies: 
        recommended_movies.append(list(content.index)[i]) 
    return recommended_movies