import pandas as pd

import pyterrier as pt
if not pt.started():
  pt.init()

df = pd.DataFrame({ 
    'docno':
    ['1', '2', '3', '4'],
    'url': 
    ['url1', 'url2', 'url2', 'url3'],
    'text': 
    ['He ran out of crashing money, so he had to stop playing',
    'The waves were crashing on the shore; it was a',
    'The waves crashing on the stock',
    'The body may perhaps compensates for the loss']
})

pd_indexer = pt.DFIndexer("./pd_index")
indexref2 = pd_indexer.index(df["text"], df["docno"], df["url"])

index = pt.IndexFactory.of(indexref2)
print(index.getCollectionStatistics().toString())

topics = pd.DataFrame([["2", "crashing"]], columns=['qid', 'query'])
BM25_br = pt.BatchRetrieve(indexref2)
res = BM25_br.transform(topics)

print(res)