# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PreProcessInput import PreProcess
from NavieBayesAlgorithm import NavieBayes
from flask import Flask,render_template,request

movies_dataframe = pd.read_csv("../output/movies.csv",header=0)
model = NavieBayes(movies_dataframe)
model.train()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend",methods = ['POST'])
def recommend():
    data = request.json
    __output = "working"
    d = dict()
    d["movieId"] = []
    d["title"] = []
    d["genres"] = []
    for row in data:
        d["movieId"].append(row[0])
        d["title"].append(row[1])
        d["genres"].append(row[2])
    test = PreProcess(d)
    df = test.apply(movies_dataframe)
    df["category"] = model.predict(df)
    frequent_category = df.groupby("category").count()["title"]
    frequent_category = frequent_category.loc[frequent_category.max()-1 >= frequent_category.values].index

    df = df.loc[df['category'].isin(frequent_category.values)]
    df = df.loc[~df['movieId'].isin(d["movieId"])]

    df = df.drop_duplicates(subset=['movieId',"title","score","rating"])
    top100_watched = df.nlargest(100,["score"])
    top10_rated = top100_watched.nlargest(10,["rating"])
    return ("\n".join(top10_rated["title"]))

# run the application
if __name__ == "__main__":
    app.run(debug=True)