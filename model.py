import pandas as pd
import numpy as np

import pickle

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

def model_start():

    #read in the initial CSV of songs and lyrics
    df = pd.read_csv('lyrics_2010.csv')

    #make a new column that combines some of the columns into other
    df['genre_combine'] = np.where(
    (df['genre'] == 'Jazz')|
    (df['genre'] == 'Indie')|
    (df['genre'] == 'R&B')|
    (df['genre'] == 'Folk')|
    (df['genre'] == 'Rock')|
    (df['genre'] == 'Electronic')|
    (df['genre'] == 'Metal')|
    (df['genre'] == 'Pop')|
    (df['genre'] == 'Not Available'), 
    'Other', 
    df['genre']
    )

    #Split the data into a test set and a train set
    train_df, test_df = train_test_split(df)

    #Using Multinomial Naive Bayes because it's a good method for text classification
    text_clf = Pipeline(
    [('vect', TfidfVectorizer()),
        ('clf', MultinomialNB(alpha=0.1))])

    # train the model
    text_clf.fit(train_df.lyrics, train_df.genre_combine)  

    # score the model
    predicted = text_clf.predict(test_df.lyrics)

    #this is the accuracy of the model
    mean1 =  np.mean(predicted == test_df.genre_combine)
    print(mean1)

    
    # #This model has better accuracy but performs worst in the actual classification
    # #Using Linear Support Vector Machines (SVM) as it's regarded as one of the best text classifier algorithms
    # from sklearn.linear_model import SGDClassifier
    # text_clf = Pipeline(
    #     [('vect', TfidfVectorizer()),
    #      ('clf', SGDClassifier()),
    #     ])


    # # train the model
    # text_clf.fit(train_df.lyrics, train_df.genre_combine)  

    # # score the model
    # predicted = text_clf.predict(test_df.lyrics)

    # mean1 = np.mean(predicted == test_df.genre_combine)
    # print(mean1)
    # Saving model to disk
    pickle.dump(text_clf, open('model.pkl','wb'))
    return 

def lyric_predict(input1):
    
    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    
    output = model.predict(input1.values())
    

        
    return output[0]
