from elasticsearch import Elasticsearch
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from search31 import *
rows = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'Documentary', 'IMAX', 'War', 'Musical', 'Western', 'Film-Noir', '(no genres listed)']

def csv_todfratings():   
    csv_df = pd.read_csv (r'ratings.csv')
    return(csv_df)
dfratings = csv_todfratings()
def csv_todfmovies():   
    csv_df = pd.read_csv (r'movies.csv')
    return(csv_df)
dfmovies = csv_todfmovies()  
def user_rated(csv_ratings,usr_id):
    rated_moviesid=[]
    test = 0
    found = 0 
    while(test == 0):
        for index, row in csv_ratings.iterrows():    
            if(row["userId"]==float(usr_id)):
                rated_moviesid.append(row["movieId"])
                rated_moviesid.append(row["rating"])
                found = 1
            if(row["userId"]!=float(usr_id) and found == 1 ):
                test = 1
                break
    return rated_moviesid

def find_genre(csv_todfmovies):
    genre_lst = {}
    for index, row in csv_todfmovies.iterrows():
        genre_lst[row["movieId"]] = row["genres"]           
    return genre_lst

def get_all(csvtodfratings):
    lst = []
    unique_list = []
    for index, row in csvtodfratings.iterrows():
        lst.append(row["userId"])
    for x in lst: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list
all_data= get_all(dfratings)
def ratingsto_genres(genre_dic,rated_lst):
    for i in range(len(rated_lst)):
        if (i%2)==0:        
            rated_lst[i] = genre_dic.get(rated_lst[i])
    return rated_lst
def mo_per_gerne(ratings_togernes):
    ratings_togernesfinal = []
    for i in range(len(ratings_togernes)):
        if (i % 2 == 0):
            if "|" in ratings_togernes[i]:
                splitgerns = ratings_togernes[i].split("|")
    
                for j in splitgerns:
                    
                    ratings_togernesfinal.append(j)
                    ratings_togernesfinal.append(ratings_togernes[i+1])
    return(ratings_togernesfinal)

def user_genrrating_df(lst):
    names_indexes =[]
    genrelst = []
    molst = []
   
    k = 0
    score = 0 
    scores =[]
    mofinal = []
    for i in range(len(lst)):
        if (i%2==0):
            names_indexes.append(lst[i])
    names_indexes = list(dict.fromkeys(names_indexes))
    for i in (names_indexes):
        k=0 
        for j in range(len(lst)) :
            if(i==lst[j]):
                k = k+1
                score = (score + lst[j+1])
                
        scores.append(i)
        scores.append(score/k)
        score = 0
    for i in range(len(scores)):
        if(i%2==0):
            genrelst.append(scores[i])
        else :
            molst.append(scores[i])
    for i in range(len(rows)):
        if rows[i] in genrelst:
            ind = genrelst.index(rows[i])
            mofinal.append(molst[ind])
        else:
            mofinal.append(np.nan)
            
    return(mofinal)

def do_forallusers():
    alldata =[]
    for i in get_all(dfratings): 
        alldata.append(user_genrrating_df(mo_per_gerne(ratingsto_genres(find_genre(dfmovies), user_rated(dfratings,i))))) 
        print(i)
    df = pd.DataFrame(alldata,index=get_all(dfratings), columns=rows)
    return df 
users_rated =  user_rated(dfratings,i)
print(lstcluster0 )

#modf = do_forallusers()
#modf.to_pickle("data.pkl")
