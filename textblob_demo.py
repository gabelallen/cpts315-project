from textblob import TextBlob

def analyze_text():
    text = input("Enter some text to analyze: ")
    blob = TextBlob(text)
    
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    print(f"\nPolarity: {polarity:.2f} (range: -1 = negative, 1 = positive)")
    print(f"Subjectivity: {subjectivity:.2f} (range: 0 = objective, 1 = subjective)")

if __name__ == "__main__":
    analyze_text()
