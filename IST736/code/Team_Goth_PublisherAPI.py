# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 01:32:51 2021

@author: shriv
"""

import requests as req
import json
from pandas.io.json import json_normalize
import pandas as pd


isbn = ['1428004858','1443220698','1153587408','1432500635','0554365464','0974878901','1150050969',
        '1419156314','1409237311','1428025693','1116968525','116146655X','1688823964', '1512121231', 
        '0890612714','1447465946', '1479377333', '9733401773', '1162660414', '111171634x', '1502733986', '1559941006', '1500496111', '1494836165', '1161467378']
h = {'Authorization': '45647_b8b9533c0c1bc5d09dfce928e4bc787f'}

df = pd.DataFrame()
for i in isbn:
    #print(i)
    resp = req.get("https://api2.isbndb.com/book/"+i, headers=h)
    jsontxt = resp.json()
    book_df = pd.DataFrame(jsontxt['book'])
    #print(book_df)
    df = df.append(book_df, ignore_index=True)
print(df)
#df.to_csv('publisher_info.csv')
df.replace('', 'N/A')

ax = df['publisher'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Books per Publisher")
ax.set_xlabel("Publisher Names")
ax.set_ylabel("Frequency")