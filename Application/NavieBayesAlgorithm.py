from sklearn.naive_bayes import GaussianNB

class NavieBayes:
    def __init__(self,dataframe):
        self.dataframe = dataframe
    
    def train(self):
        self.model = GaussianNB()
        self.model.fit(self.dataframe.drop(["movieId","title","genre","category"],axis=1), self.dataframe["category"])
    
    def predict(self,df):
        return self.model.predict(df.drop(["movieId","title","genre","category"],axis=1))