import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

def extract_words_by_pos(text, pos_tags):
    """Extract words of specific parts of speech from text."""
    blob = TextBlob(text)
    return [word for word, pos in blob.tags if pos in pos_tags]

def generate_wordclouds(video_comments, video_id, pos_tags=None):
    """Generate and save positive/negative word clouds for specific parts of speech."""
    # Separate positive and negative comments
    positive_comments = video_comments[video_comments['polarity'] > 0.1]
    negative_comments = video_comments[video_comments['polarity'] < -0.1]
    
    def generate_cloud(comments, title, filename, bg_color, colormap):
        if len(comments) > 0:
            text = " ".join(comment for comment in comments['comment_text'])
            
            # Filter words by parts of speech if pos_tags is provided
            if pos_tags:
                words = extract_words_by_pos(text, pos_tags)
                text = " ".join(words)

            custom_stopwords = {}  # add custom ignores here
            stopwords = STOPWORDS.union(custom_stopwords)  
        
            
            wordcloud = WordCloud(stopwords=STOPWORDS, width=800, height=400, 
                                  background_color=bg_color, colormap=colormap).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(title)
            plt.savefig(filename)
            plt.close()
            print(f"{title} saved as {filename}")
        else:
            print(f"No comments found for {title.lower()}")

    # Generate positive word cloud
    generate_cloud(
        positive_comments, 
        f"Positive Comments - Video {video_id}", 
        f"positive_wordcloud_{video_id}.png", 
        'white', 
        None
    )
    
    # Generate negative word cloud
    generate_cloud(
        negative_comments, 
        f"Negative Comments - Video {video_id}", 
        f"negative_wordcloud_{video_id}.png", 
        'black', 
        'Reds'
    )

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
            sentiment = TextBlob(str(comment)).sentiment  
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
    generate_wordclouds(video_comments, video_id, pos_tags=['NN', 'JJ', 'RB'])
    
    return avg_polarity, avg_subjectivity

if __name__ == "__main__":
    # Example usage
    video_id = input("Enter the video_id you want to analyze: ").strip()
    get_video_sentiment(video_id)