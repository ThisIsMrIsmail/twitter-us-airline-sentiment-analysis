import pandas as pd
import re
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Load data
dataset_file_path = "data/Tweets.csv"
df = pd.read_csv(dataset_file_path)
df['text'] = df['text'].astype(str)

# Display sample
import ace_tools_open as tools; tools.display_dataframe_to_user(name="Raw Tweets Sample", dataframe=df.head())

# 2. Lowercasing
df['lower'] = df['text'].str.lower()

# 3. Punctuation removal
df['no_punct'] = df['lower'].str.replace(r'[^\w\s]', '', regex=True)

# 4. Stop words removal (manual list)
STOPWORDS = {
    'a','an','the','and','or','but','if','while','with','is','to','for','in','on','at','by','of',
    'as','are','was','were','be','been','being','have','has','had','do','does','did','from','up',
    'down','out','so','that','this','it','its','they','them','you','your','we','us','he','she',
    'his','her','their','what','which','who','whom','where','when','how','why'
}

df['no_stop'] = df['no_punct'].apply(lambda x: " ".join(w for w in x.split() if w not in STOPWORDS))

# 5. Text standardization (slang lookup)
lookup = {'ur': 'your', 'u': 'you', 'lol': 'laugh', 'idk': 'i_do_not_know'}
df['standardized'] = df['no_stop'].apply(lambda x: " ".join(lookup.get(w, w) for w in x.split()))

# 6. (Optional) Spelling correction - skipped for performance
df['clean'] = df['standardized']

# 7. Emoji/emoticon removal (basic regex)
emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
df['no_emoji'] = df['clean'].apply(lambda x: emoji_pattern.sub(r'', x))

# 8. Tokenization (simple split)
df['tokens'] = df['no_emoji'].apply(lambda x: x.split())

# 9. Stemming
ps = PorterStemmer()
df['stemmed'] = df['tokens'].apply(lambda toks: [ps.stem(w) for w in toks])

# 10. Frequency distribution (top 20 stems)
freq = {}
for toks in df['stemmed']:
    for w in toks:
        freq[w] = freq.get(w, 0) + 1
top20 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
words, counts = zip(*top20)

plt.figure()
plt.bar(words, counts)
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Stemmed Words')
plt.tight_layout()
plt.show()

# 11. Word Cloud visualization
wc = WordCloud(width=800, height=400).generate_from_frequencies(freq)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
