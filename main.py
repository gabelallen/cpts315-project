import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from ydata_profiling import ProfileReport
import traceback
import os

# Create output directory for visualizations if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Function to add correlation coefficient to plots
def add_corr_coef(x, y, ax):
    corr, p = pearsonr(x, y)
    ax.annotate(f'r = {corr:.2f}\np = {p:.3f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, backgroundcolor='white', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# Function to analyze sentiment and engagement metrics
def analyze_sentiment_engagement(df):
    print("\nAnalyzing sentiment and engagement relationships...")
    
    # Filter out videos with fewer than 100 comments
    filtered_df = df[df['comment_total'] >= 100].copy()
    print(f"- Filtered dataset: {len(filtered_df)} videos with 100+ comments (from original {len(df)} videos)")
    
    # Define the metrics we want to analyze
    sentiment_metrics = ['avg_polarity', 'avg_subjectivity']
    engagement_metrics = ['comment_total', 'likes', 'dislikes', 'views']
    
    # 1. Create scatter plots with regression lines
    plt.figure(figsize=(20, 16))
    
    for i, sentiment in enumerate(sentiment_metrics):
        for j, engagement in enumerate(engagement_metrics):
            plt.subplot(2, 4, i*4 + j + 1)
            sns.regplot(x=sentiment, y=engagement, data=filtered_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            plt.title(f'{sentiment} vs {engagement} (100+ comments)')
            add_corr_coef(filtered_df[sentiment], filtered_df[engagement], plt.gca())
            if j == 0:  # Only add y-label for the first column
                plt.ylabel(engagement)
            if i == 1:  # Only add x-label for the bottom row
                plt.xlabel(sentiment)
    
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_engagement_scatterplots.png')
    plt.close()
    print("- Created scatter plots with regression lines")
    
    # 2. Create bins for polarity and subjectivity to see aggregate trends
    plt.figure(figsize=(20, 14))
    
    # Create bins for sentiment metrics
    filtered_df['polarity_bin'] = pd.qcut(filtered_df['avg_polarity'], 5, duplicates='drop')
    filtered_df['subjectivity_bin'] = pd.qcut(filtered_df['avg_subjectivity'], 5, duplicates='drop')
    
    # Plot aggregate metrics by polarity bins
    for i, metric in enumerate(engagement_metrics):
        plt.subplot(2, 4, i + 1)
        sns.barplot(x='polarity_bin', y=metric, data=filtered_df, estimator=np.mean, errorbar=('ci', 95))
        plt.title(f'Mean {metric} by Polarity Bin (100+ comments)')
        plt.xticks(rotation=45)
        plt.xlabel('Polarity Bin')
        plt.ylabel(f'Mean {metric}')
    
    # Plot aggregate metrics by subjectivity bins
    for i, metric in enumerate(engagement_metrics):
        plt.subplot(2, 4, i + 5)
        sns.barplot(x='subjectivity_bin', y=metric, data=filtered_df, estimator=np.mean, errorbar=('ci', 95))
        plt.title(f'Mean {metric} by Subjectivity Bin (100+ comments)')
        plt.xticks(rotation=45)
        plt.xlabel('Subjectivity Bin')
        plt.ylabel(f'Mean {metric}')
    
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_engagement_barplots.png')
    plt.close()
    print("- Created bar plots showing average engagement by sentiment bins")
    
    # 3. Create a heatmap with more detailed correlation values
    plt.figure(figsize=(12, 10))
    correlation_matrix = filtered_df[sentiment_metrics + engagement_metrics].corr()
    mask = np.zeros_like(correlation_matrix)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
    plt.title('Correlation Matrix: Sentiment vs Engagement Metrics (100+ comments)')
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_engagement_heatmap.png')
    plt.close()
    print("- Created detailed correlation heatmap")
    
    # 4. Log-transform engagement metrics (they often have right-skewed distributions)
    engagement_log = ['log_' + col for col in engagement_metrics]
    for old, new in zip(engagement_metrics, engagement_log):
        # Handle zeros with log1p
        filtered_df[new] = np.log1p(filtered_df[old].clip(lower=0))  # Using clip to handle any negative values
    
    plt.figure(figsize=(20, 16))
    for i, sentiment in enumerate(sentiment_metrics):
        for j, engagement in enumerate(engagement_log):
            plt.subplot(2, 4, i*4 + j + 1)
            sns.regplot(x=sentiment, y=engagement, data=filtered_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            plt.title(f'{sentiment} vs {engagement} (100+ comments)')
            add_corr_coef(filtered_df[sentiment], filtered_df[engagement], plt.gca())
            if j == 0:
                plt.ylabel(engagement.replace('log_', 'log('))
            if i == 1:
                plt.xlabel(sentiment)
    
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_log_engagement_scatterplots.png')
    plt.close()
    print("- Created scatter plots with log-transformed engagement metrics")
    
    # Save correlation statistics to a CSV file
    corr_results = []
    for sentiment in sentiment_metrics:
        for engagement in engagement_metrics + engagement_log:
            corr, p = pearsonr(filtered_df[sentiment], filtered_df[engagement])
            corr_results.append({
                'sentiment_metric': sentiment,
                'engagement_metric': engagement,
                'correlation': corr,
                'p_value': p,
                'significant': p < 0.05
            })
    
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv('visualizations/sentiment_engagement_correlations.csv', index=False)
    print("- Saved correlation statistics to CSV")
    
    print("\nAnalysis complete! Visualization files have been created in the 'visualizations' folder")

def main():
    # Initialize a list to track bad lines
    bad_lines = []
   
    def handle_bad_line(line):
        """Callback function to handle bad lines during CSV reading"""
        try:
            # Convert line to string if it's not already
            line_str = str(line) if not isinstance(line, str) else line
            bad_lines.append(line_str)
            return None  # Skip the bad line
        except Exception as e:
            print(f"Error processing bad line: {e}")
            return None
    try:
        print("Loading dataset...")
        # Load the dataset with error handling
        df = pd.read_csv('USvideos_with_sentiment.csv',
                        on_bad_lines=handle_bad_line,
                        engine='python')  # Using python engine for better error handling
       
        # Print summary of bad lines if any were found
        if bad_lines:
            print(f"\nWarning: {len(bad_lines)} bad lines were skipped during loading")
            print("First 3 bad lines:")
            for i, line in enumerate(bad_lines[:3]):
                print(f"{i+1}. {line}")
           
            # Save bad lines to a file with proper string handling
            try:
                with open('bad_lines.log', 'w', encoding='utf-8') as f:
                    f.write("\n".join(str(line) for line in bad_lines))
                print("\nSaved all bad lines to 'bad_lines.log'")
            except Exception as e:
                print(f"\nCould not save bad lines log: {e}")
        
        print("\nGenerating EDA report...")
        # Generate the profile report
        profile = ProfileReport(df,
                              title="YouTube Videos Dataset Profiling Report",
                              explorative=True,
                              vars={
                                  'cat': {'words': True, 'characters': False},
                                  'text': {'words': True, 'characters': False}
                              })
       
        # Save the report
        profile.to_file("youtube_videos_report.html")
        print("\nSuccessfully generated EDA report: youtube_videos_report.html")
       
        # Show basic info about the loaded data
        print("\nDataset Info:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
        
        # Run the sentiment analysis
        analyze_sentiment_engagement(df)
        
    except Exception as e:
        print("\nError occurred during processing:")
        print(traceback.format_exc())
        if bad_lines:
            print(f"\nNumber of bad lines encountered: {len(bad_lines)}")
            print("Last bad line encountered:")
            print(bad_lines[-1] if bad_lines else "None")

if __name__ == "__main__":
    main()