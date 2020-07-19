from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Function to process text in tweet
def text_processing(tweet):

    def handle_emojis(tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' POSITIVE_EMOTION ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' POSITIVE_EMOTION ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' POSITIVE_EMOTION ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' POSITIVE_EMOTION ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' NEGATIVE_EMOTION ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' NEGATIVE_EMOTION ', tweet)
        return tweet
    
    # Removing leftover punctuation
    def remove_punctuation(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    
    # Removing stopwords and unhelpful symbols
    def remove_stopwords(tweet):
        tweet_list = [element for element in tweet.split() if element != 'user']
        clean_tokens = [token for token in tweet_list if re.match(r'[^\W\d]*$', token)]
        clean_s = ' '.join(clean_tokens)
        clean_sentence = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_sentence
    no_stopwords_tweet = remove_stopwords(tweet)
    
    # Normalizing the tweets 
    def normalization(tweet_list):
        lemmatizer = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lemmatizer.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    return normalization(no_stopwords_tweet)
