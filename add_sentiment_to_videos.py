import pandas as pd
from textblob import TextBlob
from tqdm import tqdm  # For progress bar
import time

def calculate_video_sentiment(video_id, comments_df):
    """Calculate sentiment metrics for a specific video_id"""
    video_comments = comments_df[comments_df['video_id'] == video_id]
    
    if len(video_comments) == 0:
        return 0, 0, 0  # polarity, subjectivity, count
    
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
    
    return (
        video_comments['polarity'].mean(),  # avg polarity
        video_comments['subjectivity'].mean(),  # avg subjectivity
        len(video_comments)  # comment count
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
    
    # Process each video with progress bar
    for idx, row in tqdm(videos_df.iterrows(), total=len(videos_df)):
        video_id = row['video_id']
        polarity, subjectivity, count = calculate_video_sentiment(video_id, comments_df)
        
        videos_df.at[idx, 'avg_polarity'] = polarity
        videos_df.at[idx, 'avg_subjectivity'] = subjectivity
        videos_df.at[idx, 'comment_count'] = count
    
    # Save results
    output_file = 'USvideos_with_sentiment.csv'
    videos_df.to_csv(output_file, index=False)
    
    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed:.2f} seconds")
    print(f"Results saved to {output_file}")
    print(f"Videos with no comments found: {len(videos_df[videos_df['comment_count'] == 0])}")

if __name__ == "__main__":
    main()