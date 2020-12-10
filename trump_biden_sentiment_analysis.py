''' 
Sentiment Analysis of Tweets Pertaining to Donald Trump and Joe Biden

Seeing what insights can be derived from performing a sentiment analysis on Tweets 
scraped from Twitter using Snsscrape and Twitter API with keywords #DonaldTrump and 
#Trump (hashtag_donaldtrump.csv) and #JoeBiden and #Biden (hashtag_joebiden.csv).

The data files include over 1.7 million tweets collectively and were collected 
between the dates of 10/15/2020 and 11/04/2020.

For more: https://www.kaggle.com/thesammarks/2020-us-election-tweets-sentiment-analysis
'''

# numerical computation
import numpy as np

# data processing/manipulation
import pandas as pd
pd.options.mode.chained_assignment = None
import re

# data visualization
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px

# stopwords, tokenizer, stemmer
import nltk  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

# spell correction, lemmatization
from textblob import TextBlob
from textblob import Word

# sklearn
from sklearn.model_selection import train_test_split

# Load each dataset
trump_data = pd.read_csv('../input/us-election-2020-tweets/hashtag_donaldtrump.csv', lineterminator='\n')
biden_data = pd.read_csv('../input/us-election-2020-tweets/hashtag_joebiden.csv', lineterminator='\n')

# Remove unneeded columns
trump_df = trump_df.drop(columns=['tweet_id','user_id','user_name','user_screen_name',
                                  'user_description','user_join_date','collected_at'])
biden_df = biden_df.drop(columns=['tweet_id','user_id','user_name','user_screen_name',
                                  'user_description','user_join_date','collected_at'])

# Renaming columns
trump_df = trump_df.rename(columns={"likes": "Likes", "retweet_count": "Retweets", 
                                    "state": "State", "user_followers_count": "Followers"})
biden_df = biden_df.rename(columns={"likes": "Likes", "retweet_count": "Retweets", 
                                    "state": "State", "user_followers_count": "Followers"})

# Update United States country name for consistency
d = {"United States of America":"United States"}
trump_df['country'].replace(d, inplace=True)
biden_df['country'].replace(d, inplace=True)

trump_df = trump_df.loc[trump_df['country'] == "United States"]
biden_df = biden_df.loc[biden_df['country'] == "United States"]

# Drop null rows
trump_df = trump_df.dropna()
biden_df = biden_df.dropna() 
def clean_tweet(tweet, stem=False):

# -- PREPROCESS TWEETS ------------------------------------------------------------------

# Preprocess tweets
to_remove = r'\d+|http?\S+|[^A-Za-z0-9]+'
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to preprocess tweet 
def clean_tweet(tweet, stem=False, lemmatize=False):

    # Make all text lowercase
    tweet = tweet.lower()
    
    # Remove links, special characters, punctuation, numbers, etc.
    tweet = re.sub(to_remove, ' ', tweet)
        
    filtered_tweet = []
    words = word_tokenize(tweet) 

    # Remove stopwords and stem
    for word in words:
        if not word in stop_words:
            if stem:
                filtered_tweet.append(ps.stem(word))
            elif lemmatize:
                filtered_tweet.append(Word(word).lemmatize())
            else:
                filtered_tweet.append(word)
            
    return filtered_tweet


# Filtering all trump and biden tweets by applying cleantweet()
trump_data.tweet = trump_data.tweet.apply(lambda x: clean_tweet(x))
biden_data.tweet = biden_data.tweet.apply(lambda x: clean_tweet(x))

# -- SENTIMENT ANALYSIS -----------------------------------------------------------------

# Function to perform sentitment analysis on trump and biden dataframes
def sentiment_analysis(df):
    
    # Determine polarity and subjectivity
    df['Polarity'] = df['tweet'].apply(lambda x: TextBlob(' '.join(x)).sentiment.polarity)
    df['Subjectivity'] = df['tweet'].apply(lambda x: TextBlob(' '.join(x)).sentiment.subjectivity)
    
    # Classify overall sentiment
    df.loc[df.Polarity > 0,'Sentiment'] = 'positive'
    df.loc[df.Polarity == 0,'Sentiment'] = 'neutral'
    df.loc[df.Polarity < 0,'Sentiment'] = 'negative'
    
    return df[['tweet','Polarity','Subjectivity','Sentiment']].head()

# Perform sentiment analysis
sentiment_analysis(trump_df)
sentiment_analysis(biden_df)

# -- DATA VISUALIZATION -----------------------------------------------------------------

# Overall sentiment breakdown - Trump 
print("Trump Tweet Sentiment Breakdown")

trump_positive = len(trump_df.loc[trump_df.Sentiment=='positive'])
trump_neutral = len(trump_df.loc[trump_df.Sentiment=='neutral'])
trump_negative = len(trump_df.loc[trump_df.Sentiment=='negative'])

print("Number of Positive Tweets: ", trump_positive)
print("Number of Neutral Tweets: ", trump_neutral)
print("Number of Negative Tweets: ", trump_negative)

# Graphing the number of trump tweets by sentiment
data_t = {'Positive':trump_positive,'Neutral':trump_neutral,'Negative':trump_negative}
sentiment_t = list(data_t.keys()) 
num_tweets_t = list(data_t.values()) 

plt.figure(figsize = (8, 5)) 

plt.bar(sentiment_t, num_tweets_t, color ='red', width = 0.5, edgecolor='black',) 

plt.xlabel("Sentiment", fontweight ='bold') 
plt.ylabel("Number of Tweets", fontweight ='bold') 
plt.title("Trump Tweets by Sentiment", fontweight ='bold') 
plt.show() 

# Overall sentiment breakdown - Biden 
print("Biden Tweet Sentiment Breakdown")

biden_positive = len(biden_df.loc[biden_df.Sentiment=='positive'])
biden_neutral = len(biden_df.loc[biden_df.Sentiment=='neutral'])
biden_negative = len(biden_df.loc[biden_df.Sentiment=='negative'])

print("Number of Positive Tweets: ", biden_positive)
print("Number of Neutral Tweets: ", biden_neutral)
print("Number of Negative Tweets: ", biden_negative)

# Graphing the number of biden tweets by sentiment
data_b = {'Positive':biden_positive,'Neutral':biden_neutral,'Negative':biden_negative}
sentiment_b = list(data_b.keys()) 
num_tweets_b = list(data_b.values()) 

plt.figure(figsize = (8, 5)) 

plt.bar(sentiment_b, num_tweets_b, color ='blue', width = 0.5, edgecolor='black') 

plt.xlabel("Sentiment", fontweight ='bold') 
plt.ylabel("Number of Tweets", fontweight ='bold') 
plt.title("Biden Tweets by Sentiment", fontweight ='bold') 
plt.show() 

# Calculate relative percentages by sentiment - Trump
total_tweets_t = len(trump_df.Sentiment)
prop_tweets_t = list(map(lambda x: round(x/total_tweets_t,2), num_tweets_t))

# Calculate relative percentages by sentiment - Biden
total_tweets_b = len(biden_df.Sentiment)
prop_tweets_b = list(map(lambda x: round(x/total_tweets_b,2), num_tweets_b))

# Graphing relative percentages of both trump and biden tweets
bar_width = 0.25
plt.subplots(figsize=(8,8))

br1 = np.arange(3) 
br2 = [x + bar_width for x in br1] 

t = plt.bar(br1, prop_tweets_t, color ='r', width = bar_width, 
            edgecolor ='black', label ='Trump') 
b = plt.bar(br2, prop_tweets_b, color ='b', width = bar_width, 
            edgecolor ='black', label ='Biden') 
   
plt.xlabel('Sentiment',fontweight ='bold') 
plt.ylabel('Percentage of Tweets',fontweight ='bold') 
plt.xticks([r + bar_width/2 for r in range(3)],['Positive','Neutral','Negative'])
plt.legend([t,b],['Percentage of Trump Tweets','Percentage of Biden Tweets'])
plt.ylim(0.0, 1.0)
plt.title('Proportions of Tweets By Sentiment',fontweight ='bold')

plt.show()

# Function to return a string of all words in all tweets
def get_all_tweets(df,by_sentiment=False,sentiment="positive"):
    
    # Combine all words in tweets into a string
    if by_sentiment:
        if sentiment == "positive":
            words = ' '.join((df.loc[df.Sentiment=='positive'])['tweet'].apply(lambda x: ' '.join(x)))
        elif sentiment == "neutral":
            words = ' '.join((df.loc[df.Sentiment=='neutral'])['tweet'].apply(lambda x: ' '.join(x)))
        else:
            words = ' '.join((df.loc[df.Sentiment=='negative'])['tweet'].apply(lambda x: ' '.join(x)))
    else:
        words = ' '.join(df['tweet'].apply(lambda x: ' '.join(x)))
        
    return words

# Create word strings
words_trump = get_all_tweets(trump_df)
words_pos_trump = get_all_tweets(trump_df,True,"positive")
words_neu_trump = get_all_tweets(trump_df,True,"neutral")
words_neg_trump = get_all_tweets(trump_df,True,"negative")

words_biden = get_all_tweets(biden_df)
words_pos_biden = get_all_tweets(biden_df,True,"positive")
words_neu_biden = get_all_tweets(biden_df,True,"neutral")
words_neg_biden = get_all_tweets(biden_df,True,"negative")

# Tokenize word strings
tokens_trump = word_tokenize(words_trump)
tokens_pos_trump = word_tokenize(words_pos_trump)
tokens_neu_trump = word_tokenize(words_neu_trump)
tokens_neg_trump = word_tokenize(words_neg_trump)

tokens_biden = word_tokenize(words_biden)
tokens_pos_biden = word_tokenize(words_pos_biden)
tokens_neu_biden = word_tokenize(words_neu_biden)
tokens_neg_biden = word_tokenize(words_neg_biden)

# Function to plot most frquent words
def plot_word_freq(tokens,sentiment,t_or_b,color):
    fdist = FreqDist(tokens)
    fdist_df = pd.DataFrame(fdist.most_common(10), columns = ["Word","Frequency"])
    fig = px.bar(fdist_df, x="Word", y="Frequency",
                 title="<b>Most Frequently Used Words in </b>" + sentiment + " " + t_or_b + "<b>-Related Tweets</b>")
    fig.update_traces(marker=dict(color=color),selector=dict(type="bar"),
                      marker_line_color='black', marker_line_width=1.5, opacity=0.6)
    fig.show()

# Most frequent words in all trump tweets
plot_word_freq(tokens_trump,"<b>ALL</b>","<b>Trump</b>","red")
plot_word_freq(tokens_pos_trump,"<b>POSITIVE</b>","<b>Trump</b>","red")
plot_word_freq(tokens_neu_trump,"<b>NEUTRAL</b>","<b>Trump</b>","red")
plot_word_freq(tokens_neg_trump,"<b>NEGATIVE</b>","<b>Trump</b>","red")
plot_word_freq(tokens_biden,"<b>ALL</b>","<b>Biden</b>","blue")
plot_word_freq(tokens_pos_biden,"<b>POSITIVE</b>","<b>Biden</b>","blue")
plot_word_freq(tokens_neu_biden,"<b>NEUTRAL</b>","<b>Biden</b>","blue")
plot_word_freq(tokens_neg_biden,"<b>NEGATIVE</b>","<b>Biden</b>","blue")

# Function to generate word cloud
def create_wordcloud(words):
    
    # create wordcloud
    wordcloud = WordCloud(max_font_size=200, max_words=200, 
                          background_color="white").generate(words)

    # display the generated image
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

create_wordcloud(words_trump)
create_wordcloud(words_biden)

# Ploting polarity by state
fig = px.scatter(trump_df, x="State", y="Polarity", color="Polarity",
                title="<b>Trump-Related Tweet Polarity by State</b>",
                color_continuous_scale=px.colors.sequential.Inferno,
                width=1000, height=800)
fig.update_xaxes(categoryorder='category ascending')
fig.show()

# Average polarity by state (trump)
trump_state_polarity = trump_df.groupby("State",as_index=False).mean()

fig = px.bar(trump_state_polarity, x="State", y="Polarity",
            title="<b>Average Polarity of Trump-Related Tweets by State</b>")
fig.update_traces(marker=dict(color="red"),selector=dict(type="bar"),
                  marker_line_color='black', marker_line_width=0.8, opacity=0.6)
fig.show()

# Ploting polarity by state - Biden
fig = px.scatter(biden_df, x="State", y="Polarity", color="Polarity",
                title="<b>Biden-Related Tweet Polarity by State</b>",
                color_continuous_scale=px.colors.sequential.Inferno,
                width=1000, height=800)
fig.update_xaxes(categoryorder='category ascending')
fig.show()

# Average polarity by state - Biden
biden_state_polarity = biden_df.groupby("State",as_index=False).mean()

fig = px.bar(biden_state_polarity, x="State", y="Polarity",
            title="<b>Average Polarity of Biden-Related Tweets by State</b>")
fig.update_traces(marker=dict(color="blue"),selector=dict(type="bar"),
                  marker_line_color='black', marker_line_width=0.8, opacity=0.6)
fig.show()

# Polarity by Likes - Trump
fig = px.scatter(trump_df, x="Likes", y="Polarity", color="Polarity",
                title="<b>Trump-Related Tweet Polarity by Number of Likes</b>",
                color_continuous_scale=px.colors.sequential.Inferno,
                width=1000, height=800)
fig.show()

# Polarity by Likes - Biden
fig = px.scatter(biden_df, x="Likes", y="Polarity", color="Polarity",
                title="<b>Biden-Related Tweet Polarity by Number of Likes</b>",
                color_continuous_scale=px.colors.sequential.Inferno,
                width=1000, height=800)
fig.show()

# Polarity by Retweets - Trump
fig = px.scatter(trump_df, x="Retweets", y="Polarity", color="Polarity",
                title="<b>Trump-Related Tweet Polarity by Number of Retweets</b>",
                color_continuous_scale=px.colors.sequential.Inferno,
                width=1000, height=800)
fig.show()

# Polarity by Retweets - Biden
fig = px.scatter(biden_df, x="Retweets", y="Polarity", color="Polarity",
                title="<b>Biden-Related Tweet Polarity by Number of Retweets</b>",
                color_continuous_scale=px.colors.sequential.Inferno,
                width=1000, height=800)
fig.show()