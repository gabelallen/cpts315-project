import pandas as pd
from textblob import TextBlob
from tqdm import tqdm  # For progress bar
import time
import numpy as np

def calculate_video_sentiment(video_id, comments_df):
    """Calculate sentiment metrics for a specific video_id"""
    video_comments = comments_df[comments_df['video_id'] == video_id]
    
    if len(video_comments) == 0:
        return 0, 0, 0, 0  # polarity, subjectivity, count, polarization
    
    # Calculate sentiment if not already present
    if 'polarity' not in video_comments.columns:
        polarities = []
        subjectivities = []
        for comment in video_comments['comment_text']:
            try:
                sentiment = TextBlob(str(comment)).sentiment
                polarities.append(sentiment.polarity)
                subjectivities.append(sentiment.subjectivity)
            except:
                polarities.append(0)
                subjectivities.append(0)
        
        video_comments = video_comments.copy()
        video_comments['polarity'] = polarities
        video_comments['subjectivity'] = subjectivities
    
    # Calculate polarization
    def calculate_polarization(polarity_scores):
        """Calculate polarization score (0 to 1)"""
        filtered_scores = [score for score in polarity_scores if abs(score) > 0.4]
        if len(filtered_scores) < 2: #when not enough scores >0.4
            return -1
        polarization = np.std(filtered_scores)
        return polarization

    
    polarization = calculate_polarization(video_comments['polarity'])
    
    return (
        video_comments['polarity'].mean(),  # avg polarity
        video_comments['subjectivity'].mean(),  # avg subjectivity
        len(video_comments),  # comment count
        polarization  # polarization score
    )

def main():
    print("Loading data...")
    start_time = time.time()
    
    # Load videos data
    try:
        videos_df = pd.read_csv('USvideos_clean.csv', on_bad_lines='skip')
    except FileNotFoundError:
        print("Error: USvideos_clean.csv file not found.")
        return
    
    # Load comments data
    try:
        comments_df = pd.read_csv('UScomments.csv', on_bad_lines='skip')
        comments_df.dropna(subset=['comment_text'], inplace=True)
    except FileNotFoundError:
        print("Error: UScomments.csv file not found.")
        return
    
    print(f"Processing {len(videos_df)} videos...")
    
    # Prepare new columns
    videos_df['avg_polarity'] = 0.0
    videos_df['avg_subjectivity'] = 0.0
    videos_df['comment_count'] = 0
    videos_df['polarization'] = 0.0  # New column for polarization
    
    # Process each video with progress bar
    for idx, row in tqdm(videos_df.iterrows(), total=len(videos_df)):
        video_id = row['video_id']
        polarity, subjectivity, count, polarization = calculate_video_sentiment(video_id, comments_df)
        
        videos_df.at[idx, 'avg_polarity'] = polarity
        videos_df.at[idx, 'avg_subjectivity'] = subjectivity
        videos_df.at[idx, 'comment_count'] = count
        videos_df.at[idx, 'polarization'] = polarization
    
    # Save results
    output_file = 'USvideos_with_sentiment_and_polarization.csv'
    videos_df.to_csv(output_file, index=False)
    
    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed:.2f} seconds")
    print(f"Results saved to {output_file}")
    print(f"Videos with no comments found: {len(videos_df[videos_df['comment_count'] == 0])}")
    
    # Print some statistics about polarization
    if len(videos_df[videos_df['comment_count'] > 0]) > 0:
        avg_polarization = videos_df[videos_df['comment_count'] > 0]['polarization'].mean()
        print(f"\nAverage polarization score (for videos with comments): {avg_polarization:.3f}")
        print("Polarization interpretation (0-1 scale):")
        print("0.0-0.2: Very low polarization (mostly neutral/moderate opinions)")
        print("0.2-0.4: Low polarization (somewhat opinionated but not extreme)")
        print("0.4-0.6: Moderate polarization (clear opinions in both directions)")
        print("0.6-0.8: High polarization (many strong opinions)")
        print("0.8-1.0: Extreme polarization (almost all comments are strongly positive or negative)")

if __name__ == "__main__":
    main()