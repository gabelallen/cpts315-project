import pandas as pd
from ydata_profiling import ProfileReport
import traceback

#ydata_profiling EDA

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

    except Exception as e:
        print("\nError occurred during processing:")
        print(traceback.format_exc())
        if bad_lines:
            print(f"\nNumber of bad lines encountered: {len(bad_lines)}")
            print("Last bad line encountered:")
            print(bad_lines[-1] if bad_lines else "None")

if __name__ == "__main__":
    main()