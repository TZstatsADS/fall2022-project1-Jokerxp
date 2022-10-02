from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def date_to_era(date):
    if date < 0:
        return "BC300"
    elif date < 200:
        return "AD100"
    elif date < 1700:
        return "AD1600"
    elif date < 1800:
        return "AD1700"
    elif date < 1900:
        return "AD1800"
    elif date < 2000:
        return "AD1900"


def get_NRC_scores(df):
    n = len(df)
    emotions = ["fear", "anger", "anticipation", "trust", "surprise", "positive", "negative", "sadness", "disgust",
                "joy"]
    for emotion in emotions:
        df[emotion] = 0

    for i in range(n):
        nrc_obj = NRCLex(df.sentence_lowered[i])
        for k, v in nrc_obj.raw_emotion_scores.items():
            df[k][i] = v
    return df


def tokenizer(text):
    return text


def tfidfclassifier(df,stopwords):
    dates = df.original_publication_date.unique().tolist()
    all_texts = []
    for date in dates:
        df_cur = df[df.original_publication_date == date]
        text = [token for txt in df_cur.tokenized_txt for token in txt]
        all_texts.append(text)
        
    tfidf = TfidfVectorizer(tokenizer=tokenizer, lowercase=False,
                            stop_words=stopwords,smooth_idf=True,max_features=200)
    tfs = tfidf.fit_transform(all_texts)
    features = tfidf.get_feature_names()
    frequencies = tfs.todense().tolist()
    freq_table = pd.DataFrame(frequencies,columns = features)
    freq_table_category = freq_table.T.rename(columns = {i:dates[i] for i in range(len(dates))})
    freq_table_total = freq_table_category.sum(axis=1)
    freq_table_top10 = freq_table_category.sum(axis=1).sort_values(ascending=False)[:10]
    return freq_table_total, freq_table_top10, freq_table_category
    