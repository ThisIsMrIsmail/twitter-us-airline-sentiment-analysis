import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from textblob import Word
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



text=['This is introduction to NLP','It is likely to be useful, to people ','Machine learning is the new electrcity','There would be less hype around            AI and more action going forward','python is the best tool!','R is good langauage','I like this book','I want more books like this']
lookup_dict = {'nlp':'natural language processing', 'ur':'your', "wbu" : "what about you"}
text1=['I like fishing','I eat fish','There are many fishes in pound']
text2 = "What are you saying 😂. I am the boss😎, and why are you so 😒"