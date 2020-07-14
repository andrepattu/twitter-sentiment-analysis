from utils import text_processing
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# Read data using pandas
train_tweets = pd.read_csv('train_tweets.csv')
test_tweets = pd.read_csv('test_tweets.csv')

# Take a quick look at both sets of raw data
print(train_tweets.head(5))
print(test_tweets.head(5))

# Apply helper function from utils to process tweet
train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)
test_tweets['tweet_list'] = test_tweets['tweet'].apply(text_processing)

# Remove label "tweet" from positive and negative tweets
train_tweets[train_tweets['label']==1].drop('tweet',axis=1).head(5)
train_tweets[train_tweets['label']==0].drop('tweet',axis=1).head(5)

# Split data 
content_train, content_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets['label'], test_size=0.2)

#Pipeline: strings to token integer counts, integer counts to weighted TF-IDF scores and then Naive Bayes classifier 
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  
    ('tfidf', TfidfTransformer()), 
    ('classifier', MultinomialNB()),
])
pipeline.fit(content_train,label_train)

# Predict using trained model and visualize results
prediction = pipeline.predict(content_test)

print(classification_report(prediction,label_test), '\n')
print('Confusion Matrix:', '\n', confusion_matrix(prediction,label_test), '\n')
print('Prediction accuracy on test set:', accuracy_score(prediction,label_test))
