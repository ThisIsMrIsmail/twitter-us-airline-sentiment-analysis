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

# --- Download necessary NLTK data ---
# This is required for tokenization, stop words, POS tagging, and lemmatization.
nltk_resources = {
    'corpora/stopwords': 'stopwords',
    'tokenizers/punkt': 'punkt',
    'corpora/wordnet': 'wordnet',
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
}

print("--- Checking NLTK Resources ---")
for resource_path, resource_name in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
        print(f"NLTK resource '{resource_name}' found.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' not found. Downloading...")
        nltk.download(resource_name)
print("--- NLTK Resource Check Complete ---")

# --- Initialize variables ---
df_processed = None
text_column = None

# --- Load the dataset ---
print("\n--- Loading Dataset ---")
try:
    df = pd.read_csv('Tweets.csv')
    # Let's inspect the dataframe to find the text column
    print("Dataset Head:")
    print(df.head())
    # print("\nDataset Info:") # Keep this commented unless debugging, can be verbose
    # df.info()

    # Identify the text column
    if 'Tweets' in df.columns:
        text_column = 'Tweets'
    elif 'text' in df.columns:
        text_column = 'text'
    elif 'tweet' in df.columns: # from the example in chapter 2
        text_column = 'tweet'
    else:
        # If no specific column is found, try to infer or raise error
        # For now, we will raise an error if no common name is found.
        potential_text_columns = [col for col in df.columns if isinstance(df[col].iloc[0], str)]
        if not potential_text_columns:
             raise ValueError("Could not automatically identify the tweet text column. No string columns found or DataFrame is empty.")
        # If multiple string columns, this heuristic might need adjustment or explicit user input.
        # For simplicity, let's assume the first string column if not 'Tweets', 'text', or 'tweet'.
        # This is a basic heuristic and might not always be correct.
        # text_column = potential_text_columns[0]
        # print(f"Warning: Using inferred text column '{text_column}'. Please verify.")
        raise ValueError("Could not automatically identify a common tweet text column ('Tweets', 'text', 'tweet'). Please specify it in the script if it has a different name.")


    print(f"\nUsing column: '{text_column}' for tweet text.\n")
    
    df_processed = df[[text_column]].copy()
    if len(df_processed) > 1000:
        print(f"Processing a sample of 1000 tweets out of {len(df_processed)} for demonstration purposes.")
        df_processed = df_processed.head(1000)
    else:
        print(f"Processing all {len(df_processed)} tweets.")
    print("--- Dataset Loaded Successfully ---")

except FileNotFoundError:
    print("Error: Tweets.csv not found. Please ensure the file is in the correct path.")
except ValueError as ve:
    print(f"Error during data loading or column identification: {ve}")
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")

# --- Check if data loading was successful before proceeding ---
if df_processed is None or text_column is None:
    print("\nCritical Error: Data loading failed or text column not identified. Exiting script.")
    exit() # Exit if df_processed was not successfully created

# Now, df_processed and text_column are guaranteed to be defined if we reach here.

# --- Recipe 1: Lowercasing (Chapter 2, Recipe 2-1) ---
print("\nStep 1: Converting to Lowercase...")
df_processed['processed_text'] = df_processed[text_column].astype(str).apply(lambda x: x.lower())
print("Lowercase conversion example:")
print(df_processed[['processed_text']].head())

# --- Recipe 10: Dealing with Emojis (Chapter 2, Recipe 2-10) ---
print("\nStep 2: Converting Emojis to Text...")
try:
    import demoji
    try:
        # A quick test to trigger download if needed and check if it works
        demoji.replace_with_desc("test 😂", sep="_") 
    except Exception:
        print("Initial demoji check failed, attempting to download codes...")
        demoji.download_codes() # Needs to be run once if not cached

    def convert_emojis_to_text(text):
        return demoji.replace_with_desc(text, sep="_") # e.g., 😂 becomes "face_with_tears_of_joy"

    df_processed['processed_text'] = df_processed['processed_text'].apply(convert_emojis_to_text)
    print("Emoji conversion example:")
    print(df_processed[['processed_text']].head())
except ImportError:
    print("Emoji Conversion (Skipped - 'demoji' library not found). Consider installing 'demoji': pip install demoji")
except Exception as e:
    print(f"Error during emoji conversion: {e}. Skipping this step.")


# --- Recipe 2: Punctuation Removal (Chapter 2, Recipe 2-2) ---
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
print("\nStep 4: Attempting Spelling Correction (can be slow)...")
# This step can be very slow on large datasets.
# For demonstration, it's commented out by default.
# def correct_spelling(text):
#     try:
#         return str(TextBlob(text).correct())
#     except Exception as e:
#         # print(f"Error in spelling correction for text: {text[:50]}... Error: {e}")
#         return text # Return original text if correction fails
# df_processed['processed_text'] = df_processed['processed_text'].apply(correct_spelling)
# print("Spelling correction example (first few may not show changes if already correct):")
# print(df_processed[['processed_text']].head())
print("Spelling correction step is commented out by default for performance reasons.")
print("Uncomment the relevant lines in the script if you wish to run it.")


# --- Recipe 6: Tokenization (Chapter 2, Recipe 2-6) ---
print("\nStep 5: Tokenizing Text...")
df_processed['tokens'] = df_processed['processed_text'].apply(word_tokenize)
print("Tokenization example:")
print(df_processed[['tokens']].head())

# --- Recipe 3: Stop Words Removal (Chapter 2, Recipe 2-3) ---
print("\nStep 6: Removing Stop Words...")
stop_words_list = stopwords.words('english') 
stop_words = set(stop_words_list) # Convert to set for faster lookups
# Add custom stop words if needed:
# custom_stopwords = ['face', 'smiling', 'eyes', 'emoji_description_prefix'] # Example
# stop_words.update(custom_stopwords)

def remove_stopwords(tokens):
    # also remove single characters and ensure word is not just an underscore (from emoji processing)
    return [word for word in tokens if word not in stop_words and len(word) > 1 and word != '_'] 

df_processed['tokens_no_stopwords'] = df_processed['tokens'].apply(remove_stopwords)
print("Stop words removal example:")
print(df_processed[['tokens_no_stopwords']].head())

# --- Recipe 8: Lemmatization (Chapter 2, Recipe 2-8) ---
print("\nStep 7: Lemmatizing Tokens...")
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
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
            # Avoid lemmatizing if it's an empty string or just underscore after previous steps
            if word and word != '_': 
                lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
            elif word == '_': # If it's just an underscore, keep it or discard based on policy
                pass # Or lemmatized.append(word) if you want to keep it
        except Exception as e:
            # print(f"Could not lemmatize word: '{word}'. Error: {e}. Using original word.")
            if word: # Add original word if not empty
                lemmatized.append(word) 
    return lemmatized

df_processed['lemmatized_tokens'] = df_processed['tokens_no_stopwords'].apply(lemmatize_tokens)
print("Lemmatization example:")
print(df_processed[['lemmatized_tokens']].head())

# --- Final Processed Text ---
df_processed['final_processed_text'] = df_processed['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
print("\nFinal processed text example:")
print(df_processed[['final_processed_text']].head())


# --- Recipe 9: Exploratory Data Analysis (Chapter 2, Recipe 2-9) ---
print("\nStep 8: Exploratory Data Analysis on Processed Text...")

all_tokens_lemmatized = [token for sublist in df_processed['lemmatized_tokens'] for token in sublist if token] # Ensure no empty tokens

# 1. Word Frequency Distribution
if all_tokens_lemmatized:
    freq_dist = nltk.FreqDist(all_tokens_lemmatized)
    print("\nMost Common Words (Top 20):")
    # Filter out single underscores if they somehow persist and are frequent
    common_words_filtered = [(word, count) for word, count in freq_dist.most_common(30) if word != '_'][:20]
    print(common_words_filtered)

    plt.figure(figsize=(12, 6))
    # Create a new FreqDist for plotting if filtering was applied
    plot_freq_dist = nltk.FreqDist(dict(common_words_filtered))
    if plot_freq_dist:
        plot_freq_dist.plot(20, title="Top 20 Most Common Words After Preprocessing")
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show() 
    else:
        print("Frequency distribution is empty after filtering, cannot plot.")
else:
    print("No tokens to analyze for word frequency distribution.")


# 2. Word Cloud
if all_tokens_lemmatized:
    wordcloud_text = ' '.join(all_tokens_lemmatized)
    if wordcloud_text.strip() and wordcloud_text != '_': 
        print("\nGenerating Word Cloud...")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of Processed Tweets")
        plt.tight_layout()
        plt.show() 
    else:
        print("Not enough content to generate a word cloud after processing (text is empty or just underscores).")
else:
    print("No tokens to generate a word cloud.")

print("\n--- Preprocessing Complete ---")
print("The processed data is available in the 'df_processed' DataFrame.")
print("Key columns: 'lemmatized_tokens' (list of words) and 'final_processed_text' (string).")