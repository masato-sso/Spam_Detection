import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

from gensim.models import word2vec

data=pd.read_csv("./data/spam.csv",encoding="latin-1")
data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
data.rename(columns={"v1":"label","v2":"text"},inplace=True)

X=pd.DataFrame(data["text"])
Y=pd.DataFrame(data["label"])
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=10)


vec_count=CountVectorizer(min_df=3)
vec_count.fit(X_train["text"])

X_train_vec=vec_count.transform(X_train["text"])
X_test_vec=vec_count.transform(X_test["text"])

model=BernoulliNB()
model.fit(X_train_vec,Y_train["label"])

# 0.989, 0.984
# print("Train accuracy:{:.3f}".format(model.score(X_train_vec,Y_train)))
# print("Test accuracy:{:.3f}".format(model.score(X_test_vec,Y_test)))

input_list=[]
input_text=input("text->")
input_list.append(input_text)
input_data=np.array(input_list)
input_df=pd.DataFrame(input_data,columns=["text"])
input_vec=vec_count.transform(input_df["text"])

print(model.predict(input_vec))
