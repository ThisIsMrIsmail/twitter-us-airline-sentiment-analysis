import re
import pandas as pd
import string
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import Word
from autocorrect import Speller

text=['This is introduction to NLP','It is likely to be useful, to people ','Machine learning is the new electrcity','There would be less hype around            AI and more action going forward','python is the best tool!','R is good langauage','I like this book','I want more books like this']

# --------------------------------
# Data Loading
# --------------------------------
df = pd.DataFrame({"tweet": text})
print(df)

# --------------------------------
# Lowercasing
# --------------------------------
df["tweet"] = df['tweet'].str.lower()
print(df)

# --------------------------------
# Removing Punctuation
# --------------------------------
s = "I. Like. This book!"
for c in string.punctuation:
    s = s.replace(c, "")
print(s)

# --------------------------------
# Removing Stopwords
# --------------------------------
stop = stopwords.words("english")
df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x.lower() not in stop))
print(df)

# --------------------------------
# Changing Abbreviations
# --------------------------------
lookup_dict = {'nlp':'natural language processing', 'ur':'your', "wbu" : "what about you"}
def text_std(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        word = re.sub(r'[^\w\s]','',word)
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
    return " ".join(new_words)
print(text_std("I like nlp it's ur choice"))
df["standardized_tweet"] = df["tweet"].apply(text_std)
print(df['standardized_tweet'])

# --------------------------------
# Checking Spelling
# --------------------------------
spell = Speller(lang='en')
print("Correcting 'mussage':", spell('mussage'))
print("Correcting 'survice':", spell('survice'))
df['auto_tweet'] = df['tweet'].apply(spell)
print(df['auto_tweet'])

# --------------------------------
# Tokenization
# --------------------------------
df['tokens'] = df['tweet'].apply(word_tokenize)
print(df['tokens'])

# --------------------------------
# Stemming
# --------------------------------
text1=['I like fishing','I eat fish','There are many fishes in pound']
df = pd.DataFrame({"tweet": text1})
st = PorterStemmer()
df['tweet'] = df['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
print(df['tweet'])

# --------------------------------
# Lemmatization
# --------------------------------
df = pd.DataFrame({"tweet": text1})
df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['tweet'])

# --------------------------------
# Removing URLs
# --------------------------------
