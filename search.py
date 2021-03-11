from elasticsearch import Elasticsearch
import pandas as pd 
elastic_client = Elasticsearch()
search_tag = input("Enter word you want to search:")
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
dflst = []
# Pass the query dictionary to the 'body' parameter of the
# client's Search() method, and have it return results:
result = elastic_client.search(index="movies-index", body=query_body)
all_hits = result["hits"]["hits"]
if(not all_hits):print("Nothing Found")
for num, doc in enumerate(all_hits):
    # 
    for key, value in doc.items():

        if(key == "_source"):
           dflst.append((list(value.values())[1]))
      
df = pd.DataFrame(dflst,columns=['Titles']) 
print(df)
