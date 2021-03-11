from elasticsearch import Elasticsearch
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from search31 import * 
elastic_client = Elasticsearch()
search_tag = input("Enter word you want to search:")
user_id = input("Enter user id")
query_body = {
  "query": {
    "bool": {
      "must": {
        "match": {      
          "title": search_tag
        }
      }
    }
  }
}
titlelst = []
midlst = []
scorelst =[]
# Pass the query dictionary to the 'body' parameter of the
# client's Search() method, and have it return results:
def results():
    result = elastic_client.search(index="movies-index", body=query_body)
    all_hits = result["hits"]["hits"]
    if(not all_hits):print("Nothing Found")
    for num, doc in enumerate(all_hits):
        # 
        for key, value in doc.items():

            if(key == "_source"):
               midlst.append((list(value.values())[0]))
               titlelst.append((list(value.values())[1]))
            if(key == "_score"):
                scorelst.append(value)
               
            
    df = pd.DataFrame(list(zip(midlst,titlelst,scorelst)),columns=['Movieid','Titles',"Scores"])
    return(df)
def csv_todfmov():   
    csv_df = pd.read_csv (r'movies.csv')
    return(csv_df)
def csv_todf():   
    csv_df = pd.read_csv (r'ratings.csv')
    return(csv_df)
def mo_ela_ids(dfm):
    mo_ela_ids = []
    dfm.assign(Movieid=dfm.Movieid.astype(int)).sort_values(by='Movieid',inplace=True)
    mo_ela_ids = dfm["Movieid"].tolist()
    return mo_ela_ids
def mo_ela_scores(dfs):
    mo_ela_scores = []
    dfs.assign(Movieid=dfs.Movieid.astype(int)).sort_values(by='Movieid',inplace=True)
    mo_ela_scores = dfs["Scores"].tolist()
    return mo_ela_scores
def mo_movie(csv_df,df,mo_ela_ids):
    mo_lst= []
    mo= 0
    j=0
    
    for i in mo_ela_ids:#Lista Me Tous mesous orous basei ton movie id apo tin ES me tragiko search sto ratings 
        for index, row in csv_df.iterrows():
                if(row["movieId"] == float(i)) :
                    
                    j = j+1
                    mo = (mo + row["rating"])
                    
             
        mo_lst.append(mo/j)
        mo=0
        j=0
    return(mo_lst)

def usr_rating(csv_df,df,mo_ela):

    ratings =[]
    for index, row in csv_df.iterrows():
        if(row["userId"]==float(user_id)):
            for i in mo_ela :
                if(row["movieId"]==float(i)):
                    ratings.append(i)
                    ratings.append(row["rating"])            
    return ratings
def ela_title(dft):
    mo_ela_titles = []
    dft.assign(Movieid=dft.Movieid.astype(int)).sort_values(by='Movieid',inplace=True)
    mo_ela_titles = dft["Titles"].tolist()
    return mo_ela_titles
def metriki(usr_rating,mo_movie,ela_mo_score,mo_ela_ids,titlelst):
    ratings = []
    mo_ratings =[]
    for i in mo_ela_ids:
        if i in usr_rating:
            index = usr_rating.index(i)
            ratings.append(usr_rating[index+1])
        else:
            ratings.append(0)
 
    for j in range(len(mo_movie)):
       
       mo_ratings.append(mo_movie[j]+ela_mo_score[j]+ratings[j])
            
    dfr = pd.DataFrame(list(zip(mo_ela_ids,titlelst,mo_ratings)),columns=['Movieid','Titles',"Scores"])
    dfr.assign(Scores=dfr.Scores.astype(float)).sort_values(by='Scores',inplace=True)
    return dfr
def add_alldata(mo_ela_ids,csv,df): #MO new 
    genrelst =[]
    splitgenres = []
    scores = []
    score = 0 
   
    for i in mo_ela_ids:
        for index, row in csv.iterrows():
            if(row["movieId"]==float(i)):
                genrelst.append(row["genres"])
    for i in (genrelst):
        splitgenres = i.split("|")
        for j in splitgenres:
            l = 1
            score = 0 
            for userid in range(671):
                
                row = df.loc[int(userid+1)]
                score = score + row[j]
                l = l+1
        scores.append(score/l)
    return scores
x= results()


print("metriki")
print(metriki(usr_rating(csv_todf(),x,mo_ela_ids(x)),add_alldata(mo_ela_ids(x),csv_todfmov(),result),mo_ela_scores(x),mo_ela_ids(x),ela_title(x)))
#print(metriki(usr_rating(csv_todf(),x,mo_ela_ids(x)),mo_movie(csv_todf(),x,mo_ela_ids(x)),mo_ela_scores(x),mo_ela_ids(x),ela_title(x)))