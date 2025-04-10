import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def generate_wordclouds(video_comments, video_id):
    """Generate and save positive/negative word clouds"""
    # Separate positive and negative comments
    positive_comments = video_comments[video_comments['polarity'] > 0]
    negative_comments = video_comments[video_comments['polarity'] < 0]
    
    # Generate positive word cloud
    if len(positive_comments) > 0:
        positive_text = " ".join(comment for comment in positive_comments['comment_text'])
        positive_wc = WordCloud(stopwords=STOPWORDS, width=800, height=400, 
                               background_color='white').generate(positive_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(positive_wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Positive Comments - Video {video_id}")
        plt.savefig(f"positive_wordcloud_{video_id}.png")
        plt.close()
        print(f"Positive word cloud saved as positive_wordcloud_{video_id}.png")
    else:
        print("No positive comments found for word cloud")
    
    # Generate negative word cloud
    if len(negative_comments) > 0:
        negative_text = " ".join(comment for comment in negative_comments['comment_text'])
        negative_wc = WordCloud(stopwords=STOPWORDS, width=800, height=400, 
                               background_color='black', colormap='Reds').generate(negative_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(negative_wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Negative Comments - Video {video_id}")
        plt.savefig(f"negative_wordcloud_{video_id}.png")
        plt.close()
        print(f"Negative word cloud saved as negative_wordcloud_{video_id}.png")
    else:
        print("No negative comments found for word cloud")

def get_video_sentiment(video_id):
    """Main function to analyze sentiment and generate word clouds"""
    # Load the data
    try:
        df = pd.read_csv('USComments.csv', on_bad_lines='skip')
    except FileNotFoundError:
        print("Error: USComments.csv file not found.")
        return
    
    # Clean data
    df.dropna(inplace=True)
    
    # Filter comments for the specified video_id
    video_comments = df[df['video_id'] == video_id]
    
    if len(video_comments) == 0:
        print(f"No comments found for video_id: {video_id}")
        return
    
    # Calculate sentiment
    print("Calculating sentiment for comments...")
    polarity = []
    subjectivity = []
    
    for comment in video_comments['comment_text']:
        try:
            sentiment = TextBlob(str(comment)).sentiment  # str() for safety
            polarity.append(sentiment.polarity)
            subjectivity.append(sentiment.subjectivity)
        except:
            polarity.append(0)
            subjectivity.append(0)
    
    video_comments = video_comments.copy()
    video_comments['polarity'] = polarity
    video_comments['subjectivity'] = subjectivity
    
    # Calculate averages
    avg_polarity = video_comments['polarity'].mean()
    avg_subjectivity = video_comments['subjectivity'].mean()
    
    # Print results
    print(f"\nSentiment analysis for video_id: {video_id}")
    print(f"Number of comments analyzed: {len(video_comments)}")
    print(f"Average Polarity: {avg_polarity:.4f} (Range: -1 to 1)")
    print(f"Average Subjectivity: {avg_subjectivity:.4f} (Range: 0 to 1)")
    
    # Generate word clouds
    print("\nGenerating word clouds...")
    generate_wordclouds(video_comments, video_id)
    
    return avg_polarity, avg_subjectivity

if __name__ == "__main__":
    # Example usage
    video_id = input("Enter the video_id you want to analyze: ").strip()
    get_video_sentiment(video_id)