from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from itertools import chain

class PreProcess:
    def __init__(self,data):
        self.test = pd.DataFrame({
            "movieId" : np.array(data["movieId"]),
            "title" : np.array(data["title"]),
            "genres" : np.array(data["genres"])
        })
    
    def apply(self,movies_dataframe):
        self.test["genres"] = self.test["genres"].str.split("|")
        genres_lens = self.test['genres'].map(len)
        self.test = pd.DataFrame({
            'movieId': np.repeat(self.test['movieId'], genres_lens),
            'title': np.repeat(self.test['title'], genres_lens),
            'genre': chain.from_iterable(self.test['genres'])
        })
        most_frequent_genre = self.test.groupby("genre").count()["title"].max()
        most_frequent_genre = self.test.groupby("genre").count().loc[self.test.groupby("genre").count()["title"] == most_frequent_genre]
        df = movies_dataframe.loc[movies_dataframe['genre'].isin(most_frequent_genre.index)]
        genre_rating = df.groupby("genre")["rating"].mean()
        genre_rating = pd.DataFrame({
            "genre" : genre_rating.index,
            "genre_rating" :  genre_rating.values
        })

        df = pd.merge(df.drop("genre_rating",axis=1),genre_rating,on="genre")
        df["score"] = df["score"] + 1

        self.test = df

        return df