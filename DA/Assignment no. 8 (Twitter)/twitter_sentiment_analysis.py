
"""
# Importing the libraries

## For numerical calculations and data handling"""

import numpy as numpy
import pandas as pd

"""## For visualization of data in the project"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import sklearn
from sklearn.utils import shuffle 
from sklearn.feature_extraction.text import TfidfVectorizer

"""## NLP preprocessing libraries"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re
import random
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

from collections import Counter
import unicodedata as udata
import string

"""## Checking the versions"""

print(sklearn.__version__)
print(matplotlib.__version__)
print(numpy.__version__)
print(pd.__version__)
print(nltk.__version__)

"""## Reading the csv files"""

trainSet = pd.read_csv("train.csv", encoding='latin-1', header=None)
testSet = pd.read_csv("test.csv", encoding='latin-1', header=None)
trainSet

"""## Just removing the first row as it is of no use"""

trainSet = trainSet.drop([0], axis=0)
trainSet

"""## Shuffling the data in the data frame"""

trainSet = trainSet.sample(frac=1).reset_index(drop=True)

"""## Assigning names to the columns"""

trainSet.columns = ["id", "sentiment", "tweet"]

trainSet.columns

"""## Checking null values in the dataset, here we are counting null values in each column in the dataset"""

trainSet.isnull().sum()

"""## Checking the duplicates values and counting duplicates in the data set"""

trainSet.duplicated().sum()

"""### Get the first five rows from the dataset"""

trainSet.head(5)

"""## drop some unwanted column from the dataframe"""

trainSet = trainSet.drop(["id"], axis = 1)

trainSet.head(5)

"""### count the number of sentiments with respect to their tweet (0 stands for positive tweet and 1 stands for negative tweet)"""

trainSet.sentiment.value_counts()

"""## Cleaning data
add new column pre_clean_len to dataframe which is length of each tweet
"""

trainSet['pre_clean_len'] = [len(t) for t in trainSet.tweet]

"""### Finding outliers using Box plot using pre_clean_len column"""

plt.boxplot(trainSet.pre_clean_len)
fig = plt.gcf()
fig.set_size_inches(16,10)
plt.show()

"""### As there are outliers, after preprocessing, we will again test for outliers to see if we got rid of them"""

print(trainSet.shape)

"""## Cleaning Operations

#### Importing beautiful soup
#### remove @ mentions from tweets
#### remove URLs from tweets
#### converting words like isn't to is not
#### get only text from  the tweets 
#### remove utf-8-sig code
#### converting all into lower case
#### will replace non-alphabetic characters by space
#### Word Punct Tokenize and only consider words whose length is greater than 1
#### join the words
"""

import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'        # remove @ mentions from tweets
pat2 = r'https?://[^ ]+'        # remove URLs from tweets
combined_pat = r'|'.join((pat1, pat2)) #addition of pat1 and pat2
www_pat = r'www.[^ ]+'         # remove URLs from tweets
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",   # converting words like isn't to is not
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):  # define tweet_cleaner function to clean the tweets
    soup = BeautifulSoup(text, 'lxml')    # create beautiful soup object
    souped = soup.get_text()   # get only text from the tweets 
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")    # remove utf-8-sig code
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed) # calling combined_pat
    stripped = re.sub(www_pat, '', stripped) #remove URLs
    lower_case = stripped.lower()      # converting all into lower case
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case) # converting words like isn't to is not
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)       # will replace # by space
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1] # Word Punct Tokenize and only consider words whose length is greater than 1
    return (" ".join(words)).strip() # join the words

limit=31962
clean_tweet_texts = [] # initialize list
for i in range(0,limit): # batch process almost 32000 tweets 
    clean_tweet_texts.append(tweet_cleaner(trainSet['tweet'][i]))  # call tweet_cleaner function and pass parameter as all the tweets to clean the tweets and append cleaned tweets into clean_tweet_texts list

"""## clean_tweet_texts"""

nltk.download('punkt')

"""#### tokenize word in clean_tweet_texts and append it to word_tokens list"""

word_tokens = [] # initialize list for tokens
for word in clean_tweet_texts:  # for each word in clean_tweet_texts
    word_tokens.append(word_tokenize(word)) #tokenize word in clean_tweet_texts and append it to word_tokens list

"""# Lemmatizing"""

nltk.download('wordnet')

df1 = [] # initialize list df1 to store words after lemmatization
from nltk.stem import WordNetLemmatizer # import WordNetLemmatizer from nltk.stem
lemmatizer = WordNetLemmatizer() # create an object of WordNetLemmatizer
for l in word_tokens: # for loop for every tokens in word_token
    b = [lemmatizer.lemmatize(q) for q in l] #for every tokens in word_token lemmatize word and give it to b
    df1.append(b) #append b to list df1

"""# df"""

clean_df1 =[] # initialize list clean_df1 to join word tokens after lemmatization
for c in df1:  # for loop for each list in df1
    a = " ".join(c) # join words in list with space in between and give it to a
    clean_df1.append(a) # append a to clean_df1

"""# clean_df

convert clean_tweet_texts into dataframe and name it as clean_df
"""

clean_df = pd.DataFrame(clean_df1,columns=['text']) # convert clean_tweet_texts into dataframe and name it as clean_df
#clean_df['target'] = df.sentiment[:10000] # from earlier dataframe get the sentiments of each tweet and make a new column in clean_df as target and give it all the sentiment score
#clean_df

clean_df['clean_len'] = [len(t) for t in clean_df.text] # Again make a new coloumn in the dataframe and name it as clean_len which

clean_df[clean_df.clean_len > 140].head(10) # again check if any tweet is more than 140 characters

"""### No outliers anymore"""

target2 = [] # initialize list
for i in range(0,limit): # batch process 32K tweets 
    target2.append(trainSet['sentiment'][i])
clean_df['target']=target2
clean_df.head()

X = clean_df.text # get all the text in x variable
y = clean_df.target # get all the sentiments into y variable
print(X.shape) #print shape of x
print(y.shape) # print shape of y
from collections import Counter
print(set(y)) # equals to list(set(words))
print(Counter(y).values()) #

"""Remember 1 is for racist/sexist tweets and 0 is for non-racist/non-sexist tweets

# perform train and test split

X_train is the tweets of training data, X_test is the testing tweets which we have to predict, y_train is the sentiments of tweets in the traing data and y_test is the sentiments of the tweets  which we will use to measure the accuracy of the model
"""

from sklearn.model_selection  import train_test_split #from sklearn.model_selection import train_test_split to split the data into training and tesing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 1) # split the data into traing and testing set where ratio is 80:20

"""#Get Tf-idf object and save it as vect. We can select features from here we just have simply change 
#the ngram range to change the features also we can remove stop words over here with the help of stop parameter
"""

vect = TfidfVectorizer(analyzer = "word", ngram_range=(1,3))

"""# fit or training data tweets to vect

transform our training data tweets
"""

vect.fit(X_train) 
X_train_dtm = vect.transform(X_train)

"""transform our testing data tweets"""

X_test_dtm = vect.transform(X_test)

"""# Naive Bayes"""

from sklearn.naive_bayes import MultinomialNB # import Multinomial Naive Bayes model from sklearn.naive_bayes
nb = MultinomialNB(alpha = 10) # get object of Multinomial naive bayes model with alpha parameter = 10

nb.fit(X_train_dtm, y_train)# fit our both training data tweets as well as their sentiments to the multinomial naive bayes model

from sklearn.model_selection import cross_val_score  # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = nb, X = X_train_dtm, y = y_train, cv = 10) # do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation

"""predict the sentiments of testing data tweets"""

y_pred_nb = nb.predict(X_test_dtm)

"""measure the accuracy of our model on the testing data"""

from sklearn import metrics # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_nb)

"""plot the confusion matrix between our predicted sentiments and the original testing data sentiments"""

from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_nb)

"""# Nearest Neighbour"""

from sklearn.neighbors import KNeighborsClassifier  
clf_knn = KNeighborsClassifier(n_neighbors=5`)

clf_knn.fit(X_train_dtm, y_train)

from sklearn.model_selection import cross_val_score 
accuracies = cross_val_score(estimator = clf_knn, X = X_train_dtm, y = y_train, cv = 10)
accuracies.mean()

y_pred_knn = clf_knn.predict(X_test_dtm)

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_knn)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_knn)

"""# Applying the models on test.csv"""

testSet = testSet.drop([0], axis=0)

testSet

testSet.columns = ["id", "tweet"]
testSet.columns

testSet.head()

"""# Preprocessing"""

testSet.isnull().sum()

testSet.duplicated().sum()

testSet.shape

limit=17197
clean_tweet_texts2 = [] # initialize list
for i in range(1,limit): # batch process almost 32000 tweets 
    clean_tweet_texts2.append(tweet_cleaner(testSet['tweet'][i]))  # call tweet_cleaner function and pass parameter as all the tweets to clean the tweets and append cleaned tweets into clean_tweet_texts list

"""# Tokenize"""

word_tokens_2 = [] # initialize list for tokens
for word in clean_tweet_texts2:  # for each word in clean_tweet_texts
    word_tokens_2.append(word_tokenize(word)) #tokenize word in clean_tweet_texts and append it to word_tokens list

"""# Lemmatizing"""

df2 = [] # initialize list df1 to store words after lemmatization
for l in word_tokens_2: # for loop for every tokens in word_token
    b = [lemmatizer.lemmatize(q) for q in l] #for every tokens in word_token lemmatize word and give it to b
    df2.append(b) #append b to list df1

"""## df"""

clean_df2 =[] # initialize list clean_df1 to join word tokens after lemmatization
for c in df2:  # for loop for each list in df1
    a = " ".join(c) # join words in list with space in between and give it to a
    clean_df2.append(a) # append a to clean_df1

"""## clean_df"""

clean_df2 = pd.DataFrame(clean_df2,columns=['text']) # convert clean_tweet_texts into dataframe and name it as clean_df
#clean_df['target'] = df.sentiment[:10000] # from earlier dataframe get the sentiments of each tweet and make a new column in clean_df as target and give it all the sentiment score
#clean_df

clean_df2['clean_len'] = [len(t) for t in clean_df2.text] # Again make a new coloumn in the dataframe and name it as clean_len which

clean_df2[clean_df2.clean_len > 140].head(10) # check if any tweet is more than 140 characters

clean_df2.head()

clean_df2.columns

x_test = clean_df2.text
X_test_dtm = vect.transform(x_test)

prediction = nb.predict(X_test_dtm)

prediction = prediction.astype(numpy.int)

testSet['label'] = prediction

testSet.head()

submission = testSet[['id','tweet','label']]
submission.to_csv('naive_bayes_prediction.csv', index=False) # writing data to a CSV file

