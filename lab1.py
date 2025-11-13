#Author: Erin Rodriguez
#Class: CSCI 550 - AI and Cybersecurity
#Professor: Qingli Zeng
#Assignment: Lab 1 - Naive Bayes
#venv recommended. 
#pip install pandas nltk scikit-learn matplotlib
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
evalColName = 'spam'

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

#Import CountVectorizer for Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
#import TfidfTransformer to implement Tf-idf
from sklearn.feature_extraction.text import TfidfTransformer

#vectorizer = CountVectorizer()
bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1)
BOWText = bigram_vectorizer.fit_transform(df[textColName])
print("Bag of Words Applied.")
transformer = TfidfTransformer(smooth_idf=True)
tfidf = transformer.fit_transform(BOWText)
print("TF-idf applied.")

#print(tfidf.toarray()[1])

#Model Training

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(tfidf,df[evalColName],test_size=0.25, random_state=0)
mnb = MultinomialNB()
#train classifier and make predictions in 1 step
mnbTrain = mnb.fit(X_train, y_train)
y_pred = mnbTrain.predict(X_test)
y_score = mnbTrain.predict_proba(X_test)[:, 1]

#Model Evaluation
from sklearn import metrics
#Accuracy
print("Accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
#Precision
print("Precision: %s" % metrics.precision_score(y_test, y_pred))
#Recall
print(f'Recall: {metrics.recall_score(y_test, y_pred)}')
#F1-score
print(f'F1-Score: {metrics.f1_score(y_test, y_pred)}')
#AUROC (Area Under ROC Curve)
print(f'AUROC (Area Under ROC Curve): {metrics.roc_auc_score(y_test,y_score)}')
#AUPRC (Area Under Precision–Recall Curve)
print(f'AUPRC (Area Under Precision-Recall Curve): {metrics.average_precision_score(y_test, y_score)}')
#print(f'AUPRC (Areda Under Precision-Recall Curve {metrics.auc(metrics.precision_recall_curve(y_test, y_score, pos_label=1)[0], metrics.precision_recall_curve(y_test,y_score, pos_label=1)[1])}')
#Confusion Matrix
print(f'Confusion Matrix: {metrics.confusion_matrix(y_test, y_pred)}')

#Visualization
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

#Display ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
#Display Precision-Recall Curve
prec, recall, _ = metrics.precision_recall_curve(y_test, y_score)
pr_display = metrics.PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax2)
plt.show()
#Display Confusion Matrix
cm_display = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred)).plot()
plt.show()