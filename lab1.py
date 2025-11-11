#venv recommended. 
#pip install pandas nltk
#Data Preprocessing

#import pandas to read csv
import pandas as pd
#import nltk to filter stop words
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
#import nltk lemmatization functions
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

#Makes compatible with other csv files.
csvfilename = 'emails.csv'
textColName = 'text'

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatize_input(text):
    originaltk = word_tokenize(text)
    pos_tags = pos_tag(originaltk)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word,get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)

def rm_stopwords(input):
    swtokens = word_tokenize(input)
    filtered_tokens = [word for word in swtokens if word not in stop_words]
    return ' '.join(filtered_tokens)

df = pd.read_csv(csvfilename)
print("CSV '" + csvfilename + "' imported")
#convert text to all lowercase
df[textColName] = df[textColName].astype(str).str.lower()
print("Text converted to lowercase.")
#Use regex to remove punctuation, numbers, and special symbols
df[textColName] = df[textColName].str.replace(r'[^a-z ]', '', regex=True)
print("Special symbols removed.")
#remove stop words. can be done with for loop, but this way uses pandas built in feature
df[textColName] = df[textColName].apply(rm_stopwords)
print("Stop words removed.")
#apply lemmatizaton
print("Beginning Lemmatization.")
df[textColName] = df[textColName].apply(lemmatize_input)
print("Lemmatization Complete.")

#print(df)

#Feature Representation
