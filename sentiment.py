import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS

#from https://www.kaggle.com/code/ahmedhassan23mahmoud/youtube-sentiment-wordcloud-and-emojis-analysis

#discard bad lines and missing values
df = pd.read_csv('USComments.csv', on_bad_lines='skip')
print(df.shape) #removes 323

df.dropna(inplace = True)
print(df.shape) #removes 26 (691734 remain)

#calculating sentiment and adding polarity and subjectivity columns
polarity = []
subjectivity = []
line = 0
for i in df['comment_text']:
    try:
        sentiment = TextBlob(i).sentiment
        polarity.append(sentiment.polarity)
        subjectivity.append(sentiment.subjectivity)
    except:
        polarity.append(0)
        subjectivity.append(0)
        
df['polarity'] = polarity 
df['subjectivity'] = subjectivity

df.to_csv('UScomments_with_sentiment.csv')

#separate negative and positive comments
positive_polarity = df[df['polarity']==1]
negative_polarity = df[df['polarity']==-1]

#creating wordcloud for negative comments
total_negative_comments = " ".join(negative_polarity['comment_text'])
negative_wordcloud = WordCloud(stopwords=set(STOPWORDS), width=800, height=400).generate(total_negative_comments)
negative_wordcloud.to_file("negative_wordcloud.png")  # Saves to current directory
print("Negative WordCloud saved as 'negative_wordcloud.png'")

#wordcloud for positive comments
total_positive_comments=" ".join(positive_polarity['comment_text'])
positive_wordcloud = WordCloud(stopwords=set(STOPWORDS), width=800, height=400).generate(total_positive_comments)
positive_wordcloud.to_file("positive_wordcloud.png")
print("Positive WordCloud saved as 'positive_wordcloud.png'")