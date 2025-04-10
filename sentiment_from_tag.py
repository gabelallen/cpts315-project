import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

#get sentiment from a given tag. can give wordclouds for individual tags, or create a .csv file of tags with occurences >= n.

def generate_tag_wordclouds(tag_string, do_wordcloud):
    """Analyze sentiment and generate word clouds for a specific tag"""
    try:
        # Load data with explicit dtype specification
        videos_df = pd.read_csv('USvideos.csv', low_memory=False)
        comments_df = pd.read_csv('UScomments_with_sentiment.csv', low_memory=False, 
                                dtype={'video_id': str, 'comment_text': str})
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

    # Case-insensitive tag search
    try:
        pattern = re.compile(re.escape(tag_string.lower()))
        videos_df['clean_tags'] = videos_df['tags'].str.lower().fillna('')
        matching_videos = videos_df[videos_df['clean_tags'].apply(lambda x: bool(pattern.search(x)))]
    except Exception as e:
        print(f"Error processing tags: {str(e)}")
        return None

    if matching_videos.empty:
        print(f"No videos found with tag: {tag_string}")
        return None

    # Get comments for matching videos
    try:
        matching_comments = comments_df[comments_df['video_id'].isin(matching_videos['video_id'])]
    except Exception as e:
        print(f"Error matching comments: {str(e)}")
        return None
    
    if matching_comments.empty:
        print(f"No comments found for videos with tag: {tag_string}")
        return None

    # Classify comments by sentiment
    try:
        positive_comments = matching_comments[matching_comments['polarity'] > 0.1]
        negative_comments = matching_comments[matching_comments['polarity'] < -0.1]
        neutral_comments = matching_comments[(matching_comments['polarity'] >= -0.1) & 
                                          (matching_comments['polarity'] <= 0.1)]
    except Exception as e:
        print(f"Error classifying sentiment: {str(e)}")
        return None

    if do_wordcloud:
        # Generate word clouds if enough comments exist
        def create_wordcloud(comments, sentiment):
            if len(comments) < 10:
                print(f"Not enough {sentiment} comments ({len(comments)}) for word cloud")
                return None
            
            try:
                # Add tag_string to stopwords to exclude it from the word cloud
                custom_stopwords = set(STOPWORDS)
                custom_stopwords.update([tag_string.lower()])
                
                text = " ".join(comment for comment in comments['comment_text'].astype(str))
                wc = WordCloud(width=1200, height=600,
                              background_color='white' if sentiment == 'positive' else 'black',
                              colormap='viridis' if sentiment == 'positive' else 'Reds',
                              stopwords=custom_stopwords).generate(text)
                
                plt.figure(figsize=(12, 6))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"{sentiment.capitalize()} Comments for '{tag_string}' (n={len(comments)})")
                filename = f"{sentiment}_wordcloud_{tag_string[:30]}.png".replace(" ", "_").lower()
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Generated {filename}")
                return filename
            except Exception as e:
                print(f"Error generating {sentiment} word cloud: {str(e)}")
                return None

        # Create both word clouds
        try:
            create_wordcloud(positive_comments, 'positive')
            create_wordcloud(negative_comments, 'negative')
        except Exception as e:
            print(f"Error creating word clouds: {str(e)}")

    # Calculate statistics
    try:
        total_comments = len(matching_comments)
        pos_count = len(positive_comments)
        neg_count = len(negative_comments)
        neu_count = len(neutral_comments)
        
        stats = {
            'tag': tag_string,
            'total_videos': len(matching_videos),
            'total_comments': total_comments,
            'positive_comments': pos_count,
            'negative_comments': neg_count,
            'neutral_comments': neu_count,
            'avg_polarity': matching_comments['polarity'].mean(),
            'avg_subjectivity': matching_comments['subjectivity'].mean(),
            'positivity_ratio': pos_count / total_comments if total_comments > 0 else 0
        }

        # Print report with proper string formatting
        print(f"\n=== Sentiment Analysis for '{tag_string}' ===")
        print(f"Videos with this tag: {stats['total_videos']}")
        print(f"Total comments analyzed: {stats['total_comments']:,}")
        
        print("\nSentiment Distribution:")
        pos_pct = (pos_count / total_comments * 100) if total_comments > 0 else 0
        neg_pct = (neg_count / total_comments * 100) if total_comments > 0 else 0
        neu_pct = (neu_count / total_comments * 100) if total_comments > 0 else 0
        
        print(f"Positive: {pos_count:,} ({pos_pct:.1f}%)")
        print(f"Negative: {neg_count:,} ({neg_pct:.1f}%)")
        print(f"Neutral: {neu_count:,} ({neu_pct:.1f}%)")
        
        print(f"\nAverage Polarity: {stats['avg_polarity']:.3f} (1=positive, -1=negative)")
        print(f"Average Subjectivity: {stats['avg_subjectivity']:.3f} (1=subjective, 0=objective)")

        return stats
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return None

def analyze_frequent_tags(min_occurrences=5):
    """Analyze tags that appear at least min_occurrences times and save results to CSV"""
    try:
        videos_df = pd.read_csv('USvideos.csv', low_memory=False)
    except Exception as e:
        print(f"Error loading videos data: {str(e)}")
        return None

    # Extract all individual tags
    all_tags = []
    try:
        for tags in videos_df['tags'].str.lower().fillna(''):
            tags_split = [tag.strip() for tag in tags.split('|') if tag.strip()]
            all_tags.extend(tags_split)
    except Exception as e:
        print(f"Error processing tags: {str(e)}")
        return None
    
    # Count tag occurrences
    try:
        tag_counts = Counter(all_tags)
        frequent_tags = {tag: count for tag, count in tag_counts.items() if count >= min_occurrences}
    except Exception as e:
        print(f"Error counting tags: {str(e)}")
        return None
    
    if not frequent_tags:
        print(f"No tags found with at least {min_occurrences} occurrences")
        return None
    
    print(f"\nFound {len(frequent_tags)} tags with at least {min_occurrences} occurrences")
    
    # Analyze each frequent tag
    results = []
    for i, (tag, count) in enumerate(frequent_tags.items(), 1):
        print(f"\nProcessing tag {i}/{len(frequent_tags)}: {tag} (appears {count} times)")
        try:
            stats = generate_tag_wordclouds(tag, False)
            if stats is not None:
                stats['occurrences'] = count
                results.append(stats)
        except Exception as e:
            print(f"Error processing tag '{tag}': {str(e)}")
            continue
    
    if results:
        try:
            results_df = pd.DataFrame(results)
            cols = ['tag', 'occurrences'] + [col for col in results_df.columns if col not in ['tag', 'occurrences']]
            results_df = results_df[cols]
            csv_filename = f"frequent_tags_analysis_{min_occurrences}_occurrences.csv"
            results_df.to_csv(csv_filename, index=False)
            print(f"\nSaved analysis of {len(results)} tags to {csv_filename}")
            return results_df
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return None
    else:
        print("No results to save")
        return None

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Analyze a specific tag")
    print("2. Analyze all tags with at least 5 occurrences")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        tag = input("Enter a tag to analyze: ").strip()
        results = generate_tag_wordclouds(tag, True)
    elif choice == '2':
        min_occurrences = input("Enter minimum number of occurrences (default 5): ").strip()
        min_occurrences = int(min_occurrences) if min_occurrences.isdigit() else 5
        results = analyze_frequent_tags(min_occurrences)
    else:
        print("Invalid choice")