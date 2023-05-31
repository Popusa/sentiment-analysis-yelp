### IMPORTED LIBRARIES
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import string
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import simpledialog
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
### IMPORTED LIBRARIES


### GLOBAL VARIABLES
EXTENSION = ".csv"

CURRENT_PATH: Path = str(Path(__file__).parent.resolve())
### GLOBAL VARIABLES

class nlp_data_ops:
    def perform_cleaning(self,df):
        df.drop_duplicates(inplace = True)
        df.isnull().value_counts()
        df.dropna(inplace = True)
        return df
    
    def remove_specific_text(self,df):
        #ONLY CALL THIS FUNCTION IF YOU ARE USING THE "raw_yelp_review_data.csv" FILE
        df['full_review_text'] = [new_text[8:] for new_text in df['full_review_text']]
        df['full_review_text'] = [new_text.replace("check-in","") for new_text in df['full_review_text']]
        df['full_review_text'] = [new_text.lstrip('0123456789.- ') for new_text in df['full_review_text']]
        df['full_review_text'] = [new_text.lstrip('s') for new_text in df['full_review_text']]

        if type(df['star_rating'][0]) == np.int64:
            return
        else:
            df['star_rating'] = df['star_rating'].str[:2]
            df['star_rating'] = [int(rating) for rating in df['star_rating']]

    def get_pos_tag(self,tag):
        if tag.startswith('N'):
            return 'n'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('J'):
            return 'a'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'
        
    def perform_pre_processing(self,text):
        stopwords = nltk.corpus.stopwords.words('english')
        tokens = nltk.word_tokenize(text)
        lower = [word.lower() for word in tokens]
        no_stopwords = [word for word in lower if word not in stopwords]
        no_alpha = [word for word in no_stopwords if word.isalpha()]
        tokens_tagged = nltk.pos_tag(no_alpha)
        lemmatizer = nltk.WordNetLemmatizer()
        lemmatized_text = [lemmatizer.lemmatize(word[0],pos=self.get_pos_tag(word[1])) for word in tokens_tagged]
        preprocessed_text = lemmatized_text
        return ' '.join(preprocessed_text)
    
    def save_corpus_csv(self,df,filename):
        filepath = CURRENT_PATH + "\\" + filename
        df.to_csv(filepath,index=False)

    def classify_sentiment(self,score):
        if score['neg'] > score['pos']:
            return "Negative Sentiment"
        elif score['neg'] < score['pos']:
            return "Positive Sentiment"
        else:
            return "Neutral Sentiment"
        
    def extract_sent_polarity(self,score):
        return score['compound']
    
    def save_sent(self,df):
        sid = SentimentIntensityAnalyzer()
        sent_polarity_info = [sid.polarity_scores(review) for review in df['full_review_text']]
        review_sentiment = [self.classify_sentiment(scores) for scores in sent_polarity_info]
        sent_polarity = [self.extract_sent_polarity(scores) for scores in sent_polarity_info]

        df['str_sent'] = review_sentiment
        df['sent_polarity'] = sent_polarity

ROOT = tk.Tk()

ROOT.withdraw()
# the input dialog
USER_INP = simpledialog.askstring(title="Text Pre-Processor",
                                  prompt="Enter the name of the file (without the file extension) you want to pre-process:")

print("Processing Text...")
df = pd.read_csv(CURRENT_PATH + "\\" + USER_INP + EXTENSION)
pipeline = nlp_data_ops()
df = pipeline.perform_cleaning(df)
## Uncomment this line if using the "raw_yelp_review_data" file.
pipeline.remove_specific_text(df)
corpus = [pipeline.perform_pre_processing(review_corpus) for review_corpus in df['full_review_text']]
pipeline.save_sent(df)
df['full_review_text'] = corpus
pipeline.save_corpus_csv(df, USER_INP + "_processed" + EXTENSION)
print("Text Processing Complete.")
tk.messagebox.showinfo(title="Corpus Maker", message="Your text data has been transformed!")