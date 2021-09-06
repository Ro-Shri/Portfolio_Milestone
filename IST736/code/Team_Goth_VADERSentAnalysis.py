# 
# IST 736 - Text Mining 
# Final Project: the horror passage
# Function: Sentiment Analysis of Author Reviews
# Author: Becky Matthews-Pease, Rohini Shrivastava, Joyce Woznica
# Date: TBD 2021
#
###------------------------------------ Import Packages ---------------------------------------
# In this section, the packages required for the code are loaded
# These are the packages required for the entire program and no
# other imports are used later in the code

import pandas as pd
import numpy as np

# for manipulating strings
import string
# for regular expressions
import re
# after pip install clean-text
from cleantext import clean
# help with lists and tuples
from operator import itemgetter

# packages for wordclouds
# note - must install wordcloud
# conda install -c conda-forge wordcloud
import collections
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# for colors
#from colour import Color
import random
import matplotlib.colors as mcolors

# Import packages
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import *
import os, fnmatch

import random as rd

# Import sentiment intensity analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('vader_lexicon')
#-------------------------- Read in Data -------------------------------
reviewsFile = "/Users/joycewoznica/Syracuse/IST736/Project/data/AuthorReviews.csv"
reviewsDF = pd.read_csv(reviewsFile)

# initial visualizations
###------------------------------ Initial Analysis before Data Manipulation -------------------------
# reviews per author
reviewsDFbyAuthor = reviewsDF['Author'].value_counts()

# ratings per author
byAuthorbyRating = pd.crosstab(reviewsDF['Rating'], reviewsDF['Author'])
# reset index to rating level
RatingLevels = byAuthorbyRating.index.values
byAuthorbyRating['Rating Level'] = RatingLevels

# reset index
byAuthorbyRating.reset_index(inplace=True, drop=True)

# need to melt the frame
melted_byAuthorbyRating = pd.melt(byAuthorbyRating, id_vars = 'Rating Level', 
                                  var_name = 'Author', value_name = 'Number in Rating')

# set the plot size for better plotting
plt.rcParams['figure.figsize']=11,8
fg = sns.catplot(x = "Author", y = "Number in Rating", hue = "Rating Level", 
                 dodge=True, height = 6, aspect = 2, palette="Spectral", 
                 kind="bar", data=melted_byAuthorbyRating)
fg.set_xticklabels(rotation = 75, horizontalalignment = 'right', 
                         fontweight = 'light', fontsize = 'small')
fg.set(xlabel = "Author Name", ylabel = "Total Number at Rating Level", 
       title = "Total Number at each Rating Level by Author")

# clean up text for sentiment analysis
# create a function passing the dataframe and the column and run all these
# functions to clean them up
def remove_punct_more(book_string):
    # remove new lines from text
    # need to fix this one
    # need to remove illustrations
    # ** JOYCE TO TEST THIS LINE **
    book_string = book_string.replace('\[Illustration\]', '')
    # should consider removing Chapter #
    book_string = book_string.replace(r"\\t|\\n|\\r", "")
    book_string = book_string.replace("\t|\n|\r", "")
    # replace the \ in from of the apostrophes
    book_string = book_string.replace(r"[\\,]", "")
    # remove the apostrophe at the beginning on each line
    book_string = book_string.replace(r"(^\')", "")
    # remove the apostrophe at the end of each line
    book_string = book_string.replace(r"(\'$)", "")
    # remove digits (numbers) from the text
    # 3/3 - this isn't working
    #book_string = book_string.replace(r'[0-9]+', '')
    book_string = re.sub(r'\d+', '', book_string)
    # remove special (punctuation) characters
    # 3/3 - maybe fix this with individual replaces?
    spec_chars = ["!",'"',"#","%","&","(",")","_",
                  "*","+",",","-",".","/",":",";","<",
                  "=",">","?","@","[","\\","]","^","_",
                  "`","{","|","}","~","–","—","'",'“',"’",'”']
    for char in spec_chars:
        book_string = book_string.replace(char, ' ')
    # now remove the extra white space
    # ** need to fix these multiple spaces **
    #book_string = book_string.replace(' +', ' ')
    book_string = re.sub(' +', ' ', book_string)
    return book_string

## Run function to clean reviews
cleanRev = []
cleanRev = pd.DataFrame(columns = ['clean_review'])

for index, row in reviewsDF.iterrows():
    clean_review = ""
    review_string = row['Review']
    clean_review = remove_punct_more(review_string)
    cleanRev = cleanRev.append(pd.DataFrame({'clean_review': [clean_review]}))

# reset the index so matches the original restDF index 
cleanRev = cleanRev.reset_index()
cleanRev = cleanRev.drop(['index'], axis = 1)    

# to lowercase
cleanRev['clean_review']=cleanRev['clean_review'].str.lower()

# now add the text column to restDF
# okay now we can combine everything into a single dataframe
review_sentDF = pd.concat([reviewsDF, cleanRev], axis=1, sort=False)

#------------------------- Sentiment Analysis --------------------------
# Initiate sentiment intesnity analyzer and create an array to store results
sia = SIA()
results = []

for review in review_sentDF['clean_review']:
    pol_score = sia.polarity_scores(review)
    pol_score['score'] = review
    results.append(pol_score)
    
print(results[:1])

df = pd.DataFrame.from_records(results)
df_final = pd.concat([review_sentDF, df], axis=1, ignore_index=True)
df_final.columns = ['Author', 'Rating Level', 'Review', 'Clean Review', 'compound', 'neg', 'neu', 'pos', 'trash']
#df_final = df_final[['score', 'compound', 'neg', 'neu', 'pos']]
#df_final.sort_values(by='pos', ascending=False)

# create a frame that have a sentiment column that is Positive, Negative or Neutral based on the compound value
sentCol = []
sentCol = pd.DataFrame(columns = ['Sentiment'])
for index, row in df_final.iterrows():
    score = row['compound']
    if score >= 0.05:
        sent = 'Positive'
    elif (score > -0.05) and (score < 0.05):
        sent = 'Neutral'
    else:
        sent = 'Negative'
    sentCol = sentCol.append(pd.DataFrame({'Sentiment': [sent]}))           
sentCol = sentCol.reset_index()
sentCol = sentCol.drop(['index'], axis = 1)

sentDF = pd.concat([df_final, sentCol], axis=1, sort=False)
# need to get the adoption speed from classificationDF
rateCol = reviewsDF['Rating']
sentDF = pd.concat([sentDF, rateCol], axis=1, sort=False)

# Need to do something to group/get frequency of 
# AdoptionSpeed and Sentiments together with grouping
mergedDF1 = sentDF.groupby(['Rating', 'Sentiment'])
mergedDF1.groups
# need something around value_counts()
mergedDF1 = mergedDF1.size()
# This is what we need to graph by track, but need to do that stacking and reindexing
# reset index
mergedDF1 = mergedDF1.reset_index()
# rename the columns
mergedDF1.columns = ['Rating', 'Sentiment', 'Frequency']
mergedDF1 = mergedDF1.reset_index()
mergedDF1 = mergedDF1.drop(['index'], axis = 1)

### NEED AUTHOR IN HERE ****

# plot with seaborn
fg = sns.catplot(x = "Rating", y = "Frequency", hue = "Sentiment", dodge=True,
                    height = 3, aspect = 2, palette="viridis", kind="bar", data=mergedDF1)
fg.set_xticklabels(horizontalalignment = 'center', 
                         fontweight = 'light', fontsize = 'medium')
fg.set(xlabel = "Rating", ylabel = "Number of Reviews at this Rating", 
       title = "Number of Reviews by Rating showing \nSentiment of Review")

# need to make colors based on the value of compound in the data frame

sns.set_style("whitegrid")
ax = sns.catplot(x="Author", y="compound", hue = 'compound', 
                 height = 6, aspect = 2, 
                 palette="RdYlGn", jitter = 0.45, s = 8, data=sentDF)
plt.title("Sentiment of Reviews by Author", fontsize = 12)
ax._legend.remove()
plt.xlabel("Author", fontsize = 10)
plt.ylabel("Sentiment Compound Score", fontsize = 10)

### DO SOMETHING HERE with AUTHOR
custom_palette = {}
for index, row in sentDF.iterrows():
    score = row['compound']
    if score >= 0.05:
        custom_palette[score] = 'forestgreen'
    elif (score > -0.05) and (score < 0.05):
        custom_palette[score] = 'steelblue'
    else:
        custom_palette[score] = 'firebrick'

sns.set_style("whitegrid")
ax = sns.catplot(x="Author", y="compound", hue = 'compound', 
                 height = 8, aspect = 1.5, palette=custom_palette,
                 jitter = 0.45, s = 8, data=sentDF)
plt.title("Sentiment of Reviews by Author", fontsize = 12)
ax._legend.remove()
plt.xlabel("Author", fontsize = 10)
plt.ylabel("Sentiment Compound Score", fontsize = 10)
