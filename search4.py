# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:02:07 2020

@author: Nikos
"""
from elasticsearch import Elasticsearch
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import nltk
from sklearn.manifold import TSNE
from Search2 import * 
from search3 import *
from gensim.models import Word2Vec,KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
all_ratings = csv_todf()
all_movies= csv_todfmov()
movies_rated=[]
movies =[]
all_data = []
empty = []
users =[]
def clean_movietitles(doc):
    # split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens
def unique(list1): 
    x = np.array(list1) 
    return(np.unique(x)) 
for i , row in all_ratings.iterrows():
    movies_rated.append(row["movieId"])
    users.append(row["userId"])
for i,row in all_movies.iterrows():
    movies.append(row["movieId"])    
users = unique(users)
df = pd.DataFrame(empty,index=users, columns=movies)  
def create_main_df():
    for index, row in all_ratings.iterrows():       
        df.at[row["userId"],row["movieId"]]=row["rating"]
    print(df)
    df.to_pickle("data4.pkl")

df = pd.read_pickle("data4.pkl")
print(df)

def get_ratings_for_user(user_id):
    movies_peruser = df.loc[user_id]
    return movies_peruser
def get_movie_titles():
    movietitles = []
    csv_df = pd.read_csv (r'movies.csv')
    movieids = csv_df["movieId"].values
    moviegenres = csv_df["genres"]
    movietitles = csv_df["title"]
    return movieids,moviegenres,movietitles

def ntlk(lst):
    titleVer = [nltk.word_tokenize(title) for title in lst]
    return titleVer

def word_2_vec(titlesntlk):
    model = Word2Vec(titlesntlk ,min_count=1,size =19)
  
    return model
movieids , moviegenres,movietitles = get_movie_titles()

def treat_gernes():
    
    df = pd.DataFrame(list(zip(movieids, moviegenres)), 
               columns =['movieId', 'Genre']) 
    # treat null values
    df['Genre'].fillna('NA', inplace = True)
    
    # separate all genres into one list, considering comma + space as separators
    genre = df['Genre'].str.split('|').tolist()
    
    # flatten the list
    flat_genre = [item for sublist in genre for item in sublist]
    
    # convert to a set to make unique
    set_genre = set(flat_genre)
    
    # back to list
    unique_genre = list(set_genre)
    
    # remove NA\
    try:
        unique_genre.remove('NA')
    except:
        print("No NaN Values")
    # create columns by each unique genre
    df = df.reindex(df.columns.tolist() + unique_genre, axis=1, fill_value=0)
    # for each value inside column, update the dummy
    for index, row in df.iterrows():
        for val in row.Genre.split('|'):
            if val != 'NA':
                df.loc[index, val] = 1
    
    df.drop('Genre', axis = 1, inplace = True)   
    return df
df_train_2 = pd.merge(treat_gernes(),csv_todf(), on='movieId')
df_todrop = csv_todfmov()
df_todrop.drop('genres', axis=1, inplace=True)
df_names_hot=pd.merge(df_train_2,df_todrop,on="movieId")
df_names_hot.to_csv("export_dataframe.csv", index = False)
print(df_train_2[df_train_2['userId'] == 1])
y = df_names_hot.iloc[1:,1:20].values
z= word_2_vec(ntlk(movietitles))

X = np.array([z[word] for word in ntlk(movietitles)])
def add_matrixes(word2vec,onehot):
    sum_vector = []
    for i in range(0, 9065):
        sum_vector.append(onehot[i]+ word2vec[i])
    added_matrixes = np.array(sum_vector)
    return added_matrixes
