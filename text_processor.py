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