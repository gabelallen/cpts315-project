import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import numpy as np

def generate_tag_wordclouds(tag_string, do_wordcloud, split_date=None):
    """Analyze sentiment and generate word clouds for a specific tag, optionally split by date"""
    try:
        # Load data with explicit dtype specification
        videos_df = pd.read_csv('USvideos_with_sentiment.csv', low_memory=False, 
                               parse_dates=['date'])
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

    # Split by date if requested
    if split_date:
        try:
            split_date = pd.to_datetime(split_date)
            # Merge with videos to get the date for each comment
            comments_with_date = matching_comments.merge(
                videos_df[['video_id', 'date']], on='video_id', how='left')
            
            before_comments = comments_with_date[comments_with_date['date'] < split_date]
            after_comments = comments_with_date[comments_with_date['date'] >= split_date]
            
            print(f"\nSplit results by date: {split_date.date()}")
            print(f"Comments before: {len(before_comments)}")
            print(f"Comments after: {len(after_comments)}")
            
            # Analyze both periods
            stats_before = analyze_comments(before_comments, tag_string, do_wordcloud, "before")
            stats_after = analyze_comments(after_comments, tag_string, do_wordcloud, "after")
            
            if stats_before and stats_after:
                # Combine stats for return value
                combined_stats = {
                    'tag': tag_string,
                    'split_date': split_date.date(),
                    'before_total_comments': stats_before['total_comments'],
                    'after_total_comments': stats_after['total_comments'],
                    'before_avg_polarity': stats_before['avg_polarity'],
                    'after_avg_polarity': stats_after['avg_polarity'],
                    'before_polarization': stats_before['polarization'],
                    'after_polarization': stats_after['polarization'],
                    'polarity_change': stats_after['avg_polarity'] - stats_before['avg_polarity'],
                    'polarization_change': stats_after['polarization'] - stats_before['polarization'],
                    'before_avg_subjectivity': stats_before['avg_subjectivity'],
                    'after_avg_subjectivity': stats_after['avg_subjectivity']
                }
                return combined_stats
            return None
            
        except Exception as e:
            print(f"Error splitting by date: {str(e)}")
            return None
    else:
        # Regular analysis without date splitting
        return analyze_comments(matching_comments, tag_string, do_wordcloud)

def calculate_polarization(polarity_scores):
    """
    Calculate polarization score (0 to 1) where:
    0 = completely neutral/uniform opinions
    1 = completely polarized (all extreme positive or negative)
    """
    # Convert to absolute values (distance from neutral)
    abs_polarity = np.abs(polarity_scores)
    
    # Calculate polarization score (mean of absolute polarity)
    #polarization = np.mean(abs_polarity)

    # New method gets stddev instead
    filtered_scores = [score for score in polarity_scores if abs(score) > 0.4] #redefining neutral
    polarization = np.std(filtered_scores)
    
    return polarization

def plot_polarity_distribution(polarity_scores, tag_string, period_suffix=""):
    """Plot a histogram of polarity scores"""
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(polarity_scores, bins=20, range=(-1, 1), color='skyblue', edgecolor='black')
        
        title_suffix = f" ({period_suffix.replace('_', ' ')})" if period_suffix else ""
        plt.title(f"Polarity Distribution for '{tag_string}'{title_suffix} (n={len(polarity_scores)})")
        plt.xlabel("Polarity Score (-1 to 1)")
        plt.ylabel("Number of Comments")
        
        filename = f"polarity_distribution_{tag_string[:30]}{period_suffix}.png".replace(" ", "_").lower()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated polarity distribution plot: {filename}")
        return filename
    except Exception as e:
        print(f"Error generating polarity distribution plot: {str(e)}")
        return None

def plot_subjectivity_distribution(subjectivity_scores, tag_string, period_suffix=""):
    """Plot a histogram of subjectivity scores"""
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(subjectivity_scores, bins=20, range=(0, 1), color='lightgreen', edgecolor='black')
        
        title_suffix = f" ({period_suffix.replace('_', ' ')})" if period_suffix else ""
        plt.title(f"Subjectivity Distribution for '{tag_string}'{title_suffix} (n={len(subjectivity_scores)})")
        plt.xlabel("Subjectivity Score (0=Objective, 1=Subjective)")
        plt.ylabel("Number of Comments")
        
        filename = f"subjectivity_distribution_{tag_string[:30]}{period_suffix}.png".replace(" ", "_").lower()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated subjectivity distribution plot: {filename}")
        return filename
    except Exception as e:
        print(f"Error generating subjectivity distribution plot: {str(e)}")
        return None

def analyze_comments(comments, tag_string, do_wordcloud, period_suffix=""):
    """Helper function to analyze a set of comments"""
    if period_suffix:
        period_suffix = f"_{period_suffix}"
    
    try:
        positive_comments = comments[comments['polarity'] > 0.4]
        negative_comments = comments[comments['polarity'] < -0.4]
        neutral_comments = comments[(comments['polarity'] >= -0.1) & 
                                 (comments['polarity'] <= 0.1)]
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
                title_suffix = f" ({period_suffix.replace('_', ' ')})" if period_suffix else ""
                plt.title(f"{sentiment.capitalize()} Comments for '{tag_string}'{title_suffix} (n={len(comments)})")
                filename = f"{sentiment}_wordcloud_{tag_string[:30]}{period_suffix}.png".replace(" ", "_").lower()
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
        total_comments = len(comments)
        pos_count = len(positive_comments)
        neg_count = len(negative_comments)
        neu_count = len(neutral_comments)
        
        # Calculate polarization
        polarization_score = calculate_polarization(comments['polarity'])
        
        stats = {
            'tag': tag_string,
            'total_comments': total_comments,
            'positive_comments': pos_count,
            'negative_comments': neg_count,
            'neutral_comments': neu_count,
            'avg_polarity': comments['polarity'].mean(),
            'polarization': polarization_score,
            'avg_subjectivity': comments['subjectivity'].mean(),
            'positivity_ratio': pos_count / total_comments if total_comments > 0 else 0
        }

        if do_wordcloud:
            # Generate polarity and subjectivity distribution plots
            plot_polarity_distribution(comments['polarity'], tag_string, period_suffix)
            plot_subjectivity_distribution(comments['subjectivity'], tag_string, period_suffix)

        # Print report with proper string formatting
        period_label = f" ({period_suffix.replace('_', ' ')})" if period_suffix else ""
        print(f"\n=== Sentiment Analysis for '{tag_string}'{period_label} ===")
        print(f"Total comments analyzed: {stats['total_comments']:,}")
        
        print("\nSentiment Distribution:")
        pos_pct = (pos_count / total_comments * 100) if total_comments > 0 else 0
        neg_pct = (neg_count / total_comments * 100) if total_comments > 0 else 0
        neu_pct = (neu_count / total_comments * 100) if total_comments > 0 else 0
        
        print(f"Positive: {pos_count:,} ({pos_pct:.1f}%)")
        print(f"Negative: {neg_count:,} ({neg_pct:.1f}%)")
        print(f"Neutral: {neu_count:,} ({neu_pct:.1f}%)")
        
        print(f"\nAverage Polarity: {stats['avg_polarity']:.3f} (1=positive, -1=negative)")
        print(f"Polarization Score: {stats['polarization']:.3f} (0=neutral, 1=highly polarized)")
        print(f"Average Subjectivity: {stats['avg_subjectivity']:.3f} (1=subjective, 0=objective)")

        # Also categorize subjectivity
        objective_comments = len(comments[comments['subjectivity'] < 0.3])
        subjective_comments = len(comments[comments['subjectivity'] > 0.7])
        mixed_comments = total_comments - objective_comments - subjective_comments
        
        obj_pct = (objective_comments / total_comments * 100) if total_comments > 0 else 0
        subj_pct = (subjective_comments / total_comments * 100) if total_comments > 0 else 0
        mixed_pct = (mixed_comments / total_comments * 100) if total_comments > 0 else 0
        
        print("\nSubjectivity Distribution:")
        print(f"Objective: {objective_comments:,} ({obj_pct:.1f}%)")
        print(f"Mixed: {mixed_comments:,} ({mixed_pct:.1f}%)")
        print(f"Subjective: {subjective_comments:,} ({subj_pct:.1f}%)")

        return stats
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return None

def analyze_frequent_tags(min_occurrences=5, split_date=None):
    """Analyze tags that appear at least min_occurrences times and save results to CSV"""
    try:
        videos_df = pd.read_csv('USvideos_with_sentiment.csv', low_memory=False, 
                               parse_dates=['date'])
        comments_df = pd.read_csv('UScomments_with_sentiment.csv', low_memory=False, 
                                dtype={'video_id': str, 'comment_text': str})
    except Exception as e:
        print(f"Error loading data: {str(e)}")
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
            # Get videos with this tag
            pattern = re.compile(re.escape(tag.lower()))
            videos_df['clean_tags'] = videos_df['tags'].str.lower().fillna('')
            matching_videos = videos_df[videos_df['clean_tags'].apply(lambda x: bool(pattern.search(x)))]
            
            if matching_videos.empty:
                continue
                
            # Get comments for these videos
            matching_comments = comments_df[comments_df['video_id'].isin(matching_videos['video_id'])]
            
            if matching_comments.empty:
                continue
                
            # Calculate statistics
            total_comments = len(matching_comments)
            avg_polarity = matching_comments['polarity'].mean()
            polarization = calculate_polarization(matching_comments['polarity'])
            avg_subjectivity = matching_comments['subjectivity'].mean()
            
            # Count sentiment categories
            positive = len(matching_comments[matching_comments['polarity'] > 0.4])
            negative = len(matching_comments[matching_comments['polarity'] < -0.4])
            neutral = len(matching_comments[(matching_comments['polarity'] >= -0.1) & 
                                         (matching_comments['polarity'] <= 0.1)])
            
            # Count subjectivity categories
            objective = len(matching_comments[matching_comments['subjectivity'] < 0.3])
            subjective = len(matching_comments[matching_comments['subjectivity'] > 0.7])
            mixed_subj = total_comments - objective - subjective
            
            # Add to results
            results.append({
                'tag': tag,
                'occurrences': count,
                'total_comments': total_comments,
                'positive_comments': positive,
                'negative_comments': negative,
                'neutral_comments': neutral,
                'positivity_ratio': positive / total_comments if total_comments > 0 else 0,
                'negativity_ratio': negative / total_comments if total_comments > 0 else 0,
                'neutrality_ratio': neutral / total_comments if total_comments > 0 else 0,
                'objective_comments': objective,
                'subjective_comments': subjective,
                'mixed_subjectivity_comments': mixed_subj,
                'objectivity_ratio': objective / total_comments if total_comments > 0 else 0,
                'subjectivity_ratio': subjective / total_comments if total_comments > 0 else 0,
                'avg_polarity': avg_polarity,
                'polarization_score': polarization,
                'avg_subjectivity': avg_subjectivity
            })
            
            # Generate plots for tags with significant number of comments
            if total_comments >= 100:
                plot_polarity_distribution(matching_comments['polarity'], tag)
                plot_subjectivity_distribution(matching_comments['subjectivity'], tag)
            
        except Exception as e:
            print(f"Error processing tag '{tag}': {str(e)}")
            continue
    
    if results:
        try:
            results_df = pd.DataFrame(results)
            # Sort by polarization score (descending)
            results_df = results_df.sort_values('polarization_score', ascending=False)
            
            # Reorder columns for better readability
            cols = ['tag', 'occurrences', 'total_comments', 
                   'positive_comments', 'negative_comments', 'neutral_comments',
                   'positivity_ratio', 'negativity_ratio', 'neutrality_ratio',
                   'objective_comments', 'subjective_comments', 'mixed_subjectivity_comments',
                   'objectivity_ratio', 'subjectivity_ratio',
                   'avg_polarity', 'polarization_score', 'avg_subjectivity']
            
            # Only keep columns that exist in the DataFrame
            cols = [col for col in cols if col in results_df.columns]
            results_df = results_df[cols]
            
            csv_filename = f"frequent_tags_analysis_{min_occurrences}_occurrences"
            if split_date:
                csv_filename += f"_split_{split_date.replace('-', '')}"
            csv_filename += ".csv"
            
            results_df.to_csv(csv_filename, index=False)
            print(f"\nSaved analysis of {len(results)} tags to {csv_filename}")
            
            # Print top polarized tags
            print("\nTop 10 Most Polarized Tags:")
            print(results_df[['tag', 'polarization_score', 'avg_polarity', 'total_comments']]
                 .head(10).to_string(index=False))
            
            # Also show most subjective tags
            print("\nTop 10 Most Subjective Tags:")
            print(results_df[['tag', 'avg_subjectivity', 'subjectivity_ratio', 'total_comments']]
                 .sort_values('avg_subjectivity', ascending=False)
                 .head(10).to_string(index=False))
            
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
    print("2. Analyze all tags with at least n occurrences")
    print("3. Analyze with date split (before/after a specific date)")
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        tag = input("Enter a tag to analyze: ").strip()
        results = generate_tag_wordclouds(tag, True)
    elif choice == '2':
        min_occurrences = input("Enter minimum number of occurrences (default 25): ").strip()
        min_occurrences = int(min_occurrences) if min_occurrences.isdigit() else 25
        results = analyze_frequent_tags(min_occurrences)
    elif choice == '3':
        print("\nDate split analysis:")
        tag = input("Enter a tag to analyze (leave blank to analyze frequent tags): ").strip()
        split_date = input("Enter split date (YYYY-MM-DD format): ").strip()
        try:
            # Validate date format
            datetime.strptime(split_date, '%Y-%m-%d')
            if tag:
                results = generate_tag_wordclouds(tag, True, split_date)
            else:
                min_occurrences = input("Enter minimum number of occurrences (default 25): ").strip()
                min_occurrences = int(min_occurrences) if min_occurrences.isdigit() else 25
                results = analyze_frequent_tags(min_occurrences, split_date)
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
    else:
        print("Invalid choice")