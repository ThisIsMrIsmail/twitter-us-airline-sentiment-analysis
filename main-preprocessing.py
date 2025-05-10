# Import necessary libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob # For spelling correction
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# --- Installation of 'emot' library for emoji handling ---
# The 'emot' library is used in Chapter 2 for handling emojis and emoticons.
# If you haven't installed it, you might need to run:
# !pip install emot
# For this environment, we'll define the functions as described if direct install is an issue.
# However, a more common and robust library for emoji description is 'demoji'.
# For simplicity and to align with the text's approach, we'll simulate the emot functionality
# or use a regex-based approach if 'emot' is not readily available in this execution environment.

# --- Download necessary NLTK data ---
# This is required for tokenization, stop words, POS tagging, and lemmatization.
# We'll ensure each required resource is checked and downloaded if missing.
nltk_resources = {
    'corpora/stopwords': 'stopwords',
    'tokenizers/punkt': 'punkt',
    'corpora/wordnet': 'wordnet',
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
}

for resource_path, resource_name in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
        print(f"NLTK resource '{resource_name}' found.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' not found. Downloading...")
        nltk.download(resource_name)

# --- Load the dataset ---
# Assuming 'Tweets.csv' is in the accessible path.
try:
    df = pd.read_csv('Tweets.csv')
    # Let's inspect the dataframe to find the text column
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    # Based on typical CSV structures for tweets, let's assume the column is named 'Tweets' or 'text'.
    # For this example, we'll proceed assuming the column is 'Tweets'.
    # If it's different, please change 'Tweets' to the correct column name.
    if 'Tweets' in df.columns:
        text_column = 'Tweets'
    elif 'text' in df.columns:
        text_column = 'text'
    elif 'tweet' in df.columns: # from the example in chapter 2
        text_column = 'tweet'
    else:
        raise ValueError("Could not automatically identify the tweet text column. Please specify it in the script.")
    
    print(f"\nUsing column: '{text_column}' for tweet text.\n")
    # For demonstration, let's work with a sample to speed up processing,
    # especially for steps like spelling correction.
    # For the final script, we'll process a slice or the whole column if it's not too large.
    # Let's take the first 1000 tweets or all if less than 1000
    df_processed = df[[text_column]].copy()
    if len(df_processed) > 1000:
        print("Processing a sample of 1000 tweets for demonstration purposes.")
        df_processed = df_processed.head(1000)
    else:
        print(f"Processing all {len(df_processed)} tweets.")


except FileNotFoundError:
    print("Error: Tweets.csv not found. Please ensure the file is in the correct path.")
    exit()
except ValueError as ve:
    print(ve)
    exit()


# --- Recipe 1: Lowercasing (Chapter 2, Recipe 2-1) ---
# Converts all text to lowercase to ensure uniformity. "NLP" and "nlp" are treated as the same.
print("\nStep 1: Converting to Lowercase...")
df_processed['processed_text'] = df_processed[text_column].astype(str).apply(lambda x: x.lower())
print("Lowercase conversion example:")
print(df_processed[['processed_text']].head())

# --- Recipe 10: Dealing with Emojis (Chapter 2, Recipe 2-10) ---
# We'll convert emojis to their textual representation.
# The PDF mentions the 'emot' library. If not available, we can use 'demoji' or a simplified regex.
# For this script, we'll use 'demoji' as it's a common library for this.
# If demoji is not available, this step might be skipped or a placeholder shown.
try:
    import demoji
    # demoji.download_codes() # Needs to be run once if not cached.
    # It's often better to let it handle caching automatically or call it explicitly if issues persist.
    # For robustness, we can try to ensure it's downloaded if the first use fails,
    # but usually, it handles this.
    try:
        demoji.replace_with_desc("test", sep="_") # A quick test to trigger download if needed
    except Exception as e:
        print(f"Initial demoji check/download triggered: {e}")
        demoji.download_codes()

    print("\nStep 2: Converting Emojis to Text...")
    def convert_emojis_to_text(text):
        return demoji.replace_with_desc(text, sep="_") # e.g., 😂 becomes "face_with_tears_of_joy"

    df_processed['processed_text'] = df_processed['processed_text'].apply(convert_emojis_to_text)
    print("Emoji conversion example:")
    print(df_processed[['processed_text']].head())
except ImportError:
    print("\nStep 2: Converting Emojis to Text (Skipped - 'demoji' library not found)")
    print("Consider installing 'demoji': pip install demoji")
except Exception as e:
    print(f"\nStep 2: Error during emoji conversion: {e}. Skipping this step.")


# --- Recipe 2: Punctuation Removal (Chapter 2, Recipe 2-2) ---
# Removes punctuation, as it often doesn't add value and can increase data size.
# We need to be careful if emojis were converted to text like "face_with_tears_of_joy"
# The underscore should be preserved.
print("\nStep 3: Removing Punctuation...")
def remove_punctuation(text):
    # Keep alphanumeric characters, spaces, and underscores (from emoji conversion)
    text = re.sub(r'[^\w\s_]', '', text)
    # Remove extra spaces that might result
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_processed['processed_text'] = df_processed['processed_text'].apply(remove_punctuation)
print("Punctuation removal example:")
print(df_processed[['processed_text']].head())

# --- Recipe 5: Spelling Correction (Chapter 2, Recipe 2-5) ---
# Corrects common spelling mistakes. This can be computationally intensive
# and might not always be accurate for informal text like tweets (slang, abbreviations).
# Applied here before tokenization.
print("\nStep 4: Attempting Spelling Correction (can be slow)...")
# This step can be very slow on large datasets.
# Consider applying it to a smaller sample or skipping if performance is critical.
# For demonstration, we'll apply it.
# def correct_spelling(text):
# try:
# return str(TextBlob(text).correct())
# except Exception as e:
# print(f"Error in spelling correction for text: {text[:50]}... Error: {e}")
# return text # Return original text if correction fails

# df_processed['processed_text'] = df_processed['processed_text'].apply(correct_spelling)
# print("Spelling correction example (first few may not show changes if already correct):")
# print(df_processed[['processed_text']].head())
# Commenting out for speed in a general script, but the code is there.
print("Spelling correction step is commented out by default for performance reasons.")
print("Uncomment the relevant lines in the script if you wish to run it.")


# --- Recipe 6: Tokenization (Chapter 2, Recipe 2-6) ---
# Splits text into individual words (tokens).
print("\nStep 5: Tokenizing Text...")
df_processed['tokens'] = df_processed['processed_text'].apply(word_tokenize)
print("Tokenization example:")
print(df_processed[['tokens']].head())

# --- Recipe 3: Stop Words Removal (Chapter 2, Recipe 2-3) ---
# Removes common words (e.g., "is", "the", "a") that usually don't carry significant meaning.
print("\nStep 6: Removing Stop Words...")
stop_words_list = stopwords.words('english') # It's already a list, convert to set for faster lookups
stop_words = set(stop_words_list)
# Add custom stop words if needed, e.g., common words from emoji text if not desired
# custom_stopwords = ['face', 'smiling', 'eyes']
# stop_words.update(custom_stopwords)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and len(word) > 1] # also remove single characters

df_processed['tokens_no_stopwords'] = df_processed['tokens'].apply(remove_stopwords)
print("Stop words removal example:")
print(df_processed[['tokens_no_stopwords']].head())

# --- Recipe 8: Lemmatization (Chapter 2, Recipe 2-8) ---
# Reduces words to their base or dictionary form (lemma). Generally preferred over stemming.
# For better lemmatization, Part-of-Speech (POS) tagging is used.
print("\nStep 7: Lemmatizing Tokens...")
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    # Ensure averaged_perceptron_tagger is available before calling pos_tag
    # The download block at the beginning should handle this.
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # Default to noun

def lemmatize_tokens(tokens):
    lemmatized = []
    for word in tokens:
        try:
            lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        except Exception as e:
            # print(f"Could not lemmatize word: {word}. Error: {e}. Using original word.")
            lemmatized.append(word) # Fallback to original word if lemmatization fails for any reason
    return lemmatized

df_processed['lemmatized_tokens'] = df_processed['tokens_no_stopwords'].apply(lemmatize_tokens)
print("Lemmatization example:")
print(df_processed[['lemmatized_tokens']].head())

# --- Final Processed Text ---
# Join tokens back into a string for some types of EDA or if needed for certain vectorizers
df_processed['final_processed_text'] = df_processed['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
print("\nFinal processed text example:")
print(df_processed[['final_processed_text']].head())


# --- Recipe 9: Exploratory Data Analysis (Chapter 2, Recipe 2-9) ---
# Visualize the most common words after preprocessing.

print("\nStep 8: Exploratory Data Analysis on Processed Text...")

# Combine all tokens into a single list for frequency analysis
all_tokens_lemmatized = [token for sublist in df_processed['lemmatized_tokens'] for token in sublist]

# 1. Word Frequency Distribution
if all_tokens_lemmatized:
    freq_dist = nltk.FreqDist(all_tokens_lemmatized)
    print("\nMost Common Words (Top 20):")
    print(freq_dist.most_common(20))

    # Plotting the frequency distribution
    plt.figure(figsize=(12, 6))
    # Check if freq_dist is empty before plotting
    if freq_dist:
        freq_dist.plot(20, title="Top 20 Most Common Words After Preprocessing")
        plt.show() # This will attempt to display the plot. In some environments, it might save or not show.
        # plt.savefig("word_frequency_distribution.png") # Option to save the plot
        # print("Word frequency distribution plot saved as word_frequency_distribution.png")
    else:
        print("Frequency distribution is empty, cannot plot.")
else:
    print("No tokens to analyze for word frequency distribution.")


# 2. Word Cloud
if all_tokens_lemmatized:
    wordcloud_text = ' '.join(all_tokens_lemmatized)
    if wordcloud_text.strip(): # Ensure there's text to generate a cloud from
        print("\nGenerating Word Cloud...")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of Processed Tweets")
        plt.show() # This will attempt to display the plot.
        # plt.savefig("word_cloud.png") # Option to save the plot
        # print("Word cloud image saved as word_cloud.png")
    else:
        print("Not enough content to generate a word cloud after processing.")
else:
    print("No tokens to generate a word cloud.")

print("\n--- Preprocessing Complete ---")
print("The processed data is available in the 'df_processed' DataFrame.")
print("Key columns: 'lemmatized_tokens' (list of words) and 'final_processed_text' (string).")