import json
import csv
import re
from datetime import datetime

def is_alphabetical(s):
    """Check if a string contains only letters, spaces, and ampersands"""
    return bool(re.fullmatch(r'^[A-Za-z &]+$', s))

def convert_date(date_str):
    """Convert DD.MM format to datetime object (assuming year 2017)"""
    try:
        day, month = map(int, date_str.split('.'))
        return datetime(2017, month, day)
    except:
        return None  # Return None for invalid dates

# Load the category mapping from JSON
with open('US_category_id.json', 'r') as f:
    category_data = json.load(f)

# Create a dictionary mapping category IDs to category names
category_map = {}
for item in category_data['items']:
    category_map[item['id']] = item['snippet']['title']

# Read the CSV file and process data
output_rows = []
unique_categories = set()
all_categories_valid = True
date_conversion_issues = 0

with open('USvideos.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    
    for row in reader:
        # Replace category_id with category name
        category_id = row['category_id']
        if category_id in category_map:
            category_name = category_map[category_id]
            row['category_id'] = category_name
            unique_categories.add(category_name)
            
            if not is_alphabetical(category_name):
                all_categories_valid = False
                print(f"Warning: Non-alphabetical category found: {category_name}")
        else:
            row['category_id'] = 'Unknown'
            unique_categories.add('Unknown')
            all_categories_valid = False
        
        # Convert date to datetime object
        original_date = row['date']
        converted_date = convert_date(original_date)
        if converted_date is None:
            date_conversion_issues += 1
            row['date'] = ''  # Empty string for invalid dates
        else:
            row['date'] = converted_date.strftime('%Y-%m-%d')  # ISO format
            
        output_rows.append(row)

# Print verification information
print("\nVerification Results:")
print(f"Number of unique categories: {len(unique_categories)}")
print("Unique categories:", sorted(unique_categories))
print(f"All categories alphabetical: {'Yes' if all_categories_valid else 'No'}")
print(f"Date conversion issues: {date_conversion_issues}")

# Update fieldnames to ensure proper CSV output
if 'date' in fieldnames:
    # Write the updated data back to a new CSV file
    with open('USvideos_clean.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nProcessing complete. Output saved to USvideos_clean.csv")
    print("Dates have been converted to YYYY-MM-DD format (assuming year 2017).")
else:
    print("Error: 'date' field not found in the original data")