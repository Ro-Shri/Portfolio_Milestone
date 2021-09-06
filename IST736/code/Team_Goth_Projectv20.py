# 
# IST 736 - Text Mining 
# Final Project: the horror passage
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

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
# for lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
## For Stemming
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from sklearn.model_selection import train_test_split
import random as rd

# import stuff for sklearn - we use this for plotting things later
from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# for classification and prediction
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# some imports for SVM
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer

# try PCA
from sklearn.decomposition import PCA
import pylab as pl

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

###----------------------------------- Get Directories and Files ------------------------------
# directories for corpuses
blackwoodDir = '/Users/joycewoznica/Syracuse/IST736/Project/data/blackwood/'
poeDir = '/Users/joycewoznica/Syracuse/IST736/Project/data/poe/'
stokerDir = '/Users/joycewoznica/Syracuse/IST736/Project/data/stoker/'
jamesDir = '/Users/joycewoznica/Syracuse/IST736/Project/data/james/'

# the test files are in predictDir
predictDir = '/Users/joycewoznica/Syracuse/IST736/Project/data/predict/'
# list of directories
dirList = [blackwoodDir, poeDir, stokerDir, jamesDir]

# list of files for each corpus
blackwoodFileList = fnmatch.filter(os.listdir(blackwoodDir), '*.txt')
poeFileList = fnmatch.filter(os.listdir(poeDir), '*.txt')
stokerFileList = fnmatch.filter(os.listdir(stokerDir), '*.txt')
jamesFileList = fnmatch.filter(os.listdir(jamesDir), '*.txt')

# this is the test set
predictFileList = fnmatch.filter(os.listdir(predictDir), '*.txt')
#predictFileList = fnmatch.filter(os.istdir())
# list of file lists
fileLists = [blackwoodFileList, poeFileList, stokerFileList, jamesFileList]

def build_fullpath(listName, filedir):
    listName = []
    for path in os.listdir(filedir):
        full_path = os.path.join(filedir, path)
        listName.append(full_path)
    return listName

# build lists (empty) for fullpaths
blackwoodFullPath = []
poeFullPath = []
stokerFullPath = []
jamesFullPath = []
# for predicted
predictFullPath = []

# full path names
blackwoodFullPath = build_fullpath(blackwoodFileList, blackwoodDir)
poeFullPath = build_fullpath(poeFileList, poeDir)
stokerFullPath = build_fullpath(stokerFileList, stokerDir)
jamesFullPath = build_fullpath(stokerFileList, jamesDir)
# predicted
predictFullPath = build_fullpath(predictFileList, predictDir)
# remember the author is in front of the "_" for this one

# build author Dictionary
# empty dictionary
author_dict = {}
# empty dictionary
author_dict2 = {}
predict_author_dict = {}
# dictionary with integer keys
author_dict = {'blackwood': 'Algernon Blackwood',
               'poe': 'Edgar Allen Poe', 
               'stoker': 'Bram Stoker',
               'james': 'M. R. James'}

author_dict2 = {'blackwood': 'Algernon Blackwood',
               'poe': 'Edgar Allen Poe', 
               'stoker': 'Bram Stoker',
               'james': 'M. R. James',
               'all': 'Blackwood, Poe, Stoker and James',
               'predict': 'Short Story Authors'}

# need predicted author dictionary
predict_author_dict = {'blackwood': 'Algernon Blackwood',
                       'burke': 'Thomas Burke',
                       'curzon': 'George Curzon',
                       'debra': 'Lemuel De Bra',
                       'delamare': 'Walter de la Mare',
                       'doyle': 'A. Conan Doyle',
                       'golding': 'Louis Golding',
                       'hichens': 'Robert Hichens',
                       'hyne': 'Cutliffe Hyne',
                       'jacobs': 'W. W. Jacobs',
                       'lewis': 'M. G. Lewis',
                       'lynch': 'Arthur Lynch',
                       'masefield': 'John Masefield',
                       'mason': 'A. W. Mason',
                       'maugham': 'W. Somerset Maugham',
                       'mordaunt': 'Elinor Mordaunt',
                       'muir': 'Ward Muir',
                       'powys': 'T. F. Powys',
                       'pugh': 'Edwin Pugh',
                       'robertsm': 'Morley Roberts',
                       'robertsr': 'R. Ellis Roberts',
                       'stacpoole': 'H. De Vere Stacpoole',
                       'shelley': 'Mary Shelley',
                       'walpole': 'Horace Walpole',
                       'wharton': 'Edith Wharton',
                       'yeats': 'W. B. Yeats',
                       'poe': 'Edgar Allen Poe', 
                       'stoker': 'Bram Stoker',
                       'james': 'M. R. James'
                       }

# Joyce to build a book_dict as well
book_dict = {}
# dictionary with integer keys
book_dict = {
            # Blackwood Works
             '3johnsilencestories': 'Three John Silence Stories',
             '3morejohnsilencestories': 'Three More John Silence Stories',
             'aprisoneroffairyland': 'A Prisoner of Fairyland',
             'dayandnightstories': 'Day and Night Stories',
             'fourweirdtales': 'Four Weird Tales',
             'incredibleadventures': 'Incredible Adventures',
             'thebrightmessenger': 'The Bright Messenger',
             'thecentaur': 'The Centaur',
             'thedamned': 'The Damned',
             'theemptyhouseandotherghoststories': 'The Empty House and Other Ghost Stories',
             'thegardenofsurvival': 'The Garden of Survival',
             'thehumanchord': 'The Human Chord',
             'themanwhomthetreesloved': 'The Man Whom the Trees Loved',
             'thewave': 'The Wave',
             'thewendigo': 'The Wendigo',
             'thewillows': 'The Willows',
             
             # James Works
             #'athinghostandotherstories': 'A Thin Ghost and Other Stories',
             'theresidentatwhitminster': 'The Resident at Whitminster',
             'thediaryofmrpoynter': 'The Diary of Mr\. Poytner',
             'anepisodeofcathedralhistory': 'An Episode of Cathedral History',
             'thestoryofadisapperanceandanappearance': 'The Story of a Disappearance and an Appearance',
             'twodoctors': 'Two Doctors',
             #'ghoststoriesofantiquary': 'Ghost Stories of Antiquary',
             'losthearts': 'Lost Hearts',
             'countmagnus': 'Count Magnus',
             'theashtree': 'The Ash-Tree',
             'themezzotint': 'The Mezzotint',
             #'ghoststoriesofantiquarypart2': 'Ghost Stories of Antiquary Part 2',
             'aschoolstory': 'A School Story',
             'castingtherunes': 'Casting the Runes',
             'martinsclose': 'Martin\'s Close',
             'mrhumphreysandhisinheritance': 'Mr Humphreys and His Inheritance',
             'therosegarden': 'The Rose Garden',
             'thestallsofbarchestercathedral': 'The Stalls of Barchester Cathedral',
             'thetractatemiddoth': 'The Tractate Middoth',
             #
             'thefivejars': 'The Five Jars',
             #'talesofterrorandwonder': 'Tales of Terror and Wonder',
             #'theanaconda': 'The Anaconda',
             #'thebravoofvenice': 'The Bravo of Venice',
             
             # Poe Works
             'thecaskofamontillado': 'The Cask of Amontillado',
             'thefallofthehouseofusher': 'The Fall of the House of Usher',
             'themasqueofthereddeath': 'The Masque of the Red Death',
             'theraven': 'The Raven',
             #'theworksofedgarallenpoev1': 'The Works of Edgar Allen Poe Volume 1',
             'themurdersofruemorgue': 'The Murders of the Rue Morgue',
             'theovalportrait': 'The Oval Portrait',
             'theunparalleledadventuresofonehanspfaall': 'The Unparalleled Adventures of One Hans Pfaall',
             #'theworksofedgarallenpoev2': 'The Works of Edgar Allen Poe Volume 2',
             'thepitandthependulum': 'The Pit and the Pendulum',
             'thetelltaleheart': 'The Tell-Tale Heart',
             'theprematureburial': 'The Premature Burial',
             #'theworksofedgarallenpoev3': 'The Works of Edgar Allen Poe Volume 3',
             
             #'theworksofedgarallenpoev4': 'The Works of Edgar Allen Poe Volume 4',
             'theoblongbox': 'The Oblong Box',
             'thelandscapegarden': 'The Landscape Garden',
             'lossofbreath': 'Loss of Breath',
             'metzengerstein': 'Metzengerstein',             
             'thedevilinthebelfry': 'The Devil in the Belfry',
             #'theworksofedgarallenpoev5': 'The Works of Edgar Allen Poe Volume 5',
             'ataleofjerusalem': 'A Tale of Jerusalem',
             'somewordswithamummy': 'Some Words with a Mummy',
             
             # Stoker Works
             'dracula': 'Dracula',
             #'draculasguest': 'Dracula\'s Guest',
             'draculasguest': 'Dracula\'s Guest',
             'crookensands': 'Crooken Sands',
             'thejudgeshouse': 'The Judge\'s House',
             'theburialofrats': 'The Burial of Rats',
             'thecomingofabelbehenna': 'The Coming of Abel Behenna',
             'thesecretofthegrowinggold': 'The Secret of the Growing Gold',
             'thesquaw': 'The Squaw',
             'thegipsyprophecy': 'The Gipsy Prophecy',             
             #
             'lairofthewhiteworm': 'Lair of the White Worm',
             'thejewelofsevenstars': 'The Jewel of Seven Stars',
             'theladyoftheshroud': 'The Lady of the Shroud',
             'theman': 'The Man',
             'thesnakespass': 'The Snake\'s Pass',
             'themysteryofthesea': 'The Mystery of the Sea',
             
             # Predict Set of Books + one of each of 4 main authors             
             'adreamofredhands': 'A Dream of Red Hands',
             'violence': 'Violence',
             'thechinkandthechild': 'The Chink and the Child',
             'thedrumsofkairwan': 'The Drums of Kairwan',
             'alifeabowlofrice': 'A Life a Bowl of Rice',
             'thecreatures': 'The Creatures',
             'captainsharkey': 'Captain Sharkey',
             'thecallofthehand': 'The Call of the Hand',
             'frankenstein': 'Frankenstein',
             'thenomad': 'The Nomad',
             'number13': 'Number 13',
             'theransom': 'The Ransom',
             'themonkeyspaw': 'The Monkey\'s Paw',
             'themonk': 'The Monk',
             'thesentimentalmortgage': 'The Sentimental Mortgage',
             'davyjonessgift': 'Davy Jones\'s Gift',
             'hatteras': 'Hatteras',
             'thetaipan': 'The Taiapn',
             'hodge': 'Hodge',
             'therewardofenterprise': 'The Reward of Enterprise',
             'alleluia': 'Alleluia',
             'theothertwin': 'The Other Twin',
             'grearsdam': 'Grear\'s Dam',
             'thenarrowway': 'The Narrow Way',
             'thekingofmaleka': 'The King of Maleka',
             'kerfol': 'Kerfol',
             'thegoldbug': 'The Gold Bug',
             'thecrucifixionoftheoutcast': 'The Crucifixion of the Outcast',
             'thecastleofotranto': 'The Castle of Oranto'
             }

#------------------------------- Publishing Information --------------------------------------
# initialize list of lists 
authorDF = [[1, 'Edgar Allen Poe', 1809, 1849, 1827], 
            [2, 'Bram Stoker', 1847, 1912, 1897],
            [3, 'M.R. James', 1862, 1936, 1904],
            [4, 'Algernon Blackwood', 1869, 1951, 1909],
            [5, 'Thomas Burke', 1886, 1945, 1961],
            [6, 'George Curzon', 1859, 1925, 1915],
            [7, 'Lemuel De Bra', 1884, 1954, 1925],
            [8, 'Walter de la Mare', 1873, 1956, 1902],
            [9, 'A. Conan Doyle', 1859, 1930, 1892],
            [10, 'Louis Golding', 1895, 1958, 1919],
            [11, 'Robert Hichens', 1864, 1950, 1886],
            [12, 'Cutliffe Hyne', 1866, 1944, 1900],
            [13, 'W. W. Jacobs', 1863, 1943, 1885],
            [14, 'M. G. Lewis', 1775, 1818, 1796],
            [15, 'Arthur Lynch', 1861, 1934, 1893],
            [16, 'John Masefield', 1878, 1967, 1902],
            [17, 'A. W. Mason', 1865, 1948, 1895],
            [18, 'W. Somerset Maugham', 1874, 1965, 1897],
            [19, 'Elinor Mordaunt', 1872, 1942, 1902],
            [20, 'Ward Muir', 1878, 1927, 1917],
            [21, 'T. F. Powys', 1875, 1953, 1927],
            [22, 'Morley Roberts', 1857, 1942, 1887],
            [23, 'R. Ellis Roberts', 1879, 1953, 1938],
            [24, 'H. De Vere Stacpoole', 1863, 1951, 1908],
            [25, 'Mary Shelley', 1797, 1851, 1807],
            [26, 'Horace Walpole', 1717, 1797, 1764],
            [27, 'Edith Wharton', 1862, 1937, 1905],
            [28, 'W. B. Yeats', 1865, 1939, 1886]]

# Create the pandas DataFrame 
authorDF = pd.DataFrame(authorDF, columns = ['Row', 'Author', 'BirthYear', 'DeathYear', 'FirstPubYear']) 

#------------------------------- Publishing TimeLine ------------------------------------------
author_names = authorDF['Author']
order = []
plt.rcParams['figure.figsize']=11,8
ax = plt.gca()
for index, author in authorDF.iterrows():
    x_vals = []
    y_vals = []
    y_vals = [author['Row'], author['Row'], author['Row']]
    x_vals = [author['BirthYear'], author['FirstPubYear'], author['DeathYear']]
    mymark = 'o'
    myline = 'solid'
    mylabel = author['Author']
    
    if mylabel == 'Edgar Allen Poe' or mylabel == 'Bram Stoker' or mylabel == 'M.R. James' or mylabel == 'Algernon Blackwood':
        mymark = 's'
        myline = 'dashed'

    plt.plot(x_vals, y_vals, marker = mymark, label = mylabel, linestyle = myline)
    o = 27 - index
    order = order + [o]
    
plt.title("Author Birthdate, First Publication Date, Date of Death", fontsize = 12)
plt.xlabel("Dates", fontsize = 10)
handles, labels = ax.get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
            shadow=True, fancybox=True, 
            loc='right', bbox_to_anchor = (1.3, 0.5)) 
ax.axes.yaxis.set_visible(False)
ax.set_xlim(1705, 1975)
plt.show()

#-------------------------------- Clean the data ----------------------------------------------
# get rid of headers and footers that are gutenberg specific
# first need to remove all the header and footer standard gutenbert stuff - should be able to
# do with line numbers
#------------------------------

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

# Read in the text files for each book selected
# empty DF of the author and each book
def read_book (author, fullPathList):
    bookDF = pd.DataFrame(columns = ['book_author', 'book_name', 'book_text'])
    # we can put the author in now and the book name as we process this
    for book in fullPathList:
        # extract the book name
        temp = book.split('/')
        fname = temp[len(temp)-1]
        book_name = fname.split('.')[0]
        # open book for reading
        book_text = open(book, 'r')
        # get raw data
        book_raw = book_text.read()
        # get number of lines
        book_lines = book_raw.split('\n')
        # could use this
        # kill the end of the book (last 365)
        # 3/2 - need to modify this to remove everything above this line
        # *** START OF THIS PROJECT GUTENBERG EBOOK ... up to the \n
        book_lines = book_lines[0:len(book_lines)-365]
        # kill the first few lines of the book (first 30)
        # 3/2 - need to modify this to remove everything below this line
        # *** END OF THIS PROJECT GUTENBERG EBOOK ... to \n
        book_lines = book_lines[27:len(book_lines)]
        # put it all back as a single text object
        # using list comprehension 
        blinesToStr = ' '.join([str(elem) for elem in book_lines]) 
        # send it for cleaning as a big chunk
        clean_bookText = remove_punct_more(blinesToStr)
        # close the book
        book_text.close()
        # now need to add the booktext to the dataframe
        df_row = {'book_author': author,
                  'book_name': book_name,
                  'book_text': clean_bookText}
        # append the row to the bookDF
        bookDF = bookDF.append(df_row, ignore_index = True)
    return bookDF
        
# this should return a dataframe of the author, book name and cleaned text for CountVectorizer
# Now call for each author
# blackwood, poe, stoker, james

# Blackwood
blackwoodDF = []
blackwoodDF = read_book('blackwood', blackwoodFullPath)

# Poe
poeDF = []
poeDF = read_book('poe', poeFullPath)
# issue with the Raven - but works for Poe - need to extract the allocolades to Poe
# what to do with the volumes texts

# Stoker
stokerDF = []
stokerDF = read_book('stoker', stokerFullPath)

# James
jamesDF = []
jamesDF = read_book('james', jamesFullPath)

# special read for the predict to split out the book and author name 
# can combine the functions passing 'test' or 'train', but for now
# two functions
def read_predict_book (fullPathList):
    bookDF = pd.DataFrame(columns = ['book_author', 'book_name', 'book_text'])
    # we can put the author in now and the book name as we process this
    for book in fullPathList:
        #print('Book is ', book)
        # extract the book name
        temp = book.split('/')
        fname = temp[len(temp)-1]
        author = fname.split('_')[0]
        temp = fname.split('_')[1]
        book_name = temp.split('.')[0]
        # open book for reading
        book_text = open(book, 'r')
        # get raw data
        book_raw = book_text.read()
        # get number of lines
        book_lines = book_raw.split('\n')
        # could use this
        # kill the end of the book (last 365)
        book_lines = book_lines[0:len(book_lines)-365]
        # kill the first few lines of the book (first 30)
        book_lines = book_lines[30:len(book_lines)]
        # put it all back as a single text object
        # using list comprehension 
        blinesToStr = ' '.join([str(elem) for elem in book_lines]) 
        # send it for cleaning as a big chunk
        clean_bookText = remove_punct_more(blinesToStr)
        # close the book
        book_text.close()
        # now need to add the booktext to the dataframe
        df_row = {'book_author': author,
                  'book_name': book_name,
                  'book_text': clean_bookText}
        # append the row to the bookDF
        bookDF = bookDF.append(df_row, ignore_index = True)
    return bookDF

predictDF = []
predictDF = read_predict_book(predictFullPath)

# join all frames together for joint vectorization
# build a single DF for all the four source authors
frames = [blackwoodDF, poeDF, stokerDF, jamesDF]
allframes = [blackwoodDF, poeDF, stokerDF, jamesDF, predictDF]
bookDF = pd.concat(frames)
allbookDF = pd.concat(allframes)
# reindex
bookDF.reset_index(inplace=True, drop=True)
allbookDF.reset_index(inplace=True, drop=True)

#-------------------------------- Vectorization for MultinomialNB ---------------------------
# ** NOTE: this is using non-stemmed data **
# Create an instance of CountVectorizer (one for the corpus)
# now that we have good, clean data
blackwoodCV = CountVectorizer(input = 'content', 
                              analyzer = 'word',
                              stop_words='english')
poeCV = CountVectorizer(input = 'conent', 
                        analyzer = 'word',
                        stop_words='english')
stokerCV = CountVectorizer(input = 'content',
                           analyzer = 'word',
                           stop_words='english')
jamesCV = CountVectorizer(input = 'content',
                          analyzer = 'word',
                          stop_words = 'english')
bookCV = CountVectorizer(input = 'content',
                         analyzer = 'word',
                         stop_words = 'english')
predictCV = CountVectorizer(input = 'content',
                            analyzer = 'word',
                            stop_words = 'english')
allbookCV = CountVectorizer(input = 'content',
                            analyzer = 'word',
                            stop_words = 'english')

# add a stemmer
STEMMER=PorterStemmer()
#print(STEMMER.stem("fishings"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):   #I like dogs a lot111 !!"
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()  
    words = [STEMMER.stem(w) for w in words]
    return words

allbookCV_STEM = CountVectorizer(input = 'content',
                                analyzer = 'word',
                                tokenizer = MY_STEMMER,
                                stop_words = 'english')


# build total list of just the book text
blackwood_text_list = blackwoodDF['book_text'].tolist()
poe_text_list = poeDF['book_text'].tolist()
stoker_text_list = stokerDF['book_text'].tolist()
james_text_list = jamesDF['book_text'].tolist()
book_text_list = bookDF['book_text'].tolist()
predict_text_list = predictDF['book_text'].tolist()
allbook_text_list = allbookDF['book_text'].tolist()

# execute the transform and then get the feature names (the word - vocabulary)
blackwoodDTM = blackwoodCV.fit_transform(blackwood_text_list)
poeDTM = poeCV.fit_transform(poe_text_list)
stokerDTM = stokerCV.fit_transform(stoker_text_list)
jamesDTM = jamesCV.fit_transform(james_text_list)
bookDTM = bookCV.fit_transform(book_text_list)
predictDTM = predictCV.fit_transform(predict_text_list)
allbookDTM = allbookCV.fit_transform(allbook_text_list)
allbookDTM_STEM = allbookCV_STEM.fit_transform(allbook_text_list)

# create the vocabulary list by running feature_names - features being words
blackwoodVocab = blackwoodCV.get_feature_names()
poeVocab = poeCV.get_feature_names()
stokerVocab = stokerCV.get_feature_names()
jamesVocab = jamesCV.get_feature_names()
bookVocab = bookCV.get_feature_names()
predictVocab = predictCV.get_feature_names()
allbookVocab = allbookCV.get_feature_names()
allbookVocab_STEM = allbookCV_STEM.get_feature_names()

# finally put these in a dataframe - one for each corpus
# this is what is used for test_train
blackwoodVectorDF = pd.DataFrame(blackwoodDTM.toarray(), columns = blackwoodVocab)
poeVectorDF = pd.DataFrame(poeDTM.toarray(), columns = poeVocab)
stokerVectorDF = pd.DataFrame(stokerDTM.toarray(), columns = stokerVocab)
jamesVectorDF = pd.DataFrame(jamesDTM.toarray(), columns = jamesVocab)
bookVectorDF = pd.DataFrame(bookDTM.toarray(), columns = bookVocab)
predictVectorDF = pd.DataFrame(predictDTM.toarray(), columns = predictVocab)
allbookVectorDF = pd.DataFrame(allbookDTM.toarray(), columns = allbookVocab)
allbookVectorDF_STEM = pd.DataFrame(allbookDTM_STEM.toarray(), columns = allbookVocab_STEM)

# now we need to "attach" the author to each frame
blackwoodLabel = blackwoodDF['book_author']
# array for author labels
blackwoodLabelArray = blackwoodLabel.to_numpy()
poeLabel = poeDF['book_author']
poeLabelArray = poeLabel.to_numpy()
stokerLabel = stokerDF['book_author']
stokerLabelArray = stokerLabel.to_numpy()
jamesLabel = jamesDF['book_author']
jamesLabelArray = jamesLabel.to_numpy()
bookLabel = bookDF['book_author']
bookLabelArray = bookLabel.to_numpy()
predictLabel = predictDF['book_author']
predictLabelArray = predictLabel.to_numpy()
allbookLabel = allbookDF['book_author']
allbookLabelArray = allbookLabel.to_numpy()

# make copies of the final DF, so we can add the label column
blackwoodLabelVectorDF = blackwoodVectorDF.copy()
poeLabelVectorDF = poeVectorDF.copy()
stokerLabelVectorDF = stokerVectorDF.copy()
jamesLabelVectorDF = jamesVectorDF.copy()
bookLabelVectorDF = bookVectorDF.copy()
predictLabelVectorDF = predictVectorDF.copy()
allbookLabelVectorDF = allbookVectorDF.copy()
allbookLabelVectorDF_STEM = allbookVectorDF_STEM.copy()

# insert label into position 0 (first) to *each* dataframe
# this is the *CountVectorizer* vector - no Binary, no TFIDF
blackwoodLabelVectorDF.insert(0, 'book_author', blackwoodLabel)
poeLabelVectorDF.insert(0, 'book_author', poeLabel)
stokerLabelVectorDF.insert(0, 'book_author', stokerLabel)
jamesLabelVectorDF.insert(0, 'book_author', jamesLabel)
bookLabelVectorDF.insert(0, 'book_author', bookLabel)
predictLabelVectorDF.insert(0, 'book_author', predictLabel)
allbookLabelVectorDF.insert(0, 'book_author', allbookLabel)
allbookLabelVectorDF_STEM.insert(0, 'book_author', allbookLabel)

###------------------------------------- Word Cloud Analysis -----------------------------------------
# Using the vocabulary list - do a wordcloud
# first we need to somehow rank the words by most common
# ranking of the words

# set stopwords
my_stopwords = set(nltk.corpus.stopwords.words('english'))
my_stopwords.update('gutenberg')

def plot_wordcloud (vocab, author, numwords):
    author_name = author_dict2.get(author)
    graph_title = "Top " + str(numwords) + " (less stopwords) Most Common Words in " + author_name
    # remove only standard 192 stop words
    wordcloud_text = WordCloud(stopwords=my_stopwords, collocations=False, background_color="black", 
                               colormap = 'RdGy',
                               prefer_horizontal = 0.85,
                               max_font_size= 30, max_words=numwords).generate(' '.join(vocab))
    # show the plot
    plt.figure(figsize = (15,15))
    plt.axis("off")
    plt.imshow(wordcloud_text, interpolation='bilinear')
    plt.title(graph_title, fontsize = 16)
    plt.show()

# code from this source: 
# https://kavita-ganesan.com/how-to-use-countvectorizer/#Using-CountVectorizer-to-Extract-N-Gram-Term-Counts
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """return n-gram counts in descending order of counts"""
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    results=[]
    # word index, count i
    for idx, count in sorted_items:
        # get the ngram name
        n_gram=feature_names[idx]
        # collect as a list of tuples
        results.append((n_gram,count))
    return results

# Use the functions
# Blackwood Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(blackwoodVocab, 
                                               sort_coo(blackwoodDTM[0].tocoo()),300)))
plot_wordcloud(words_only, 'blackwood', 250)

# Poe Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(poeVocab, 
                                               sort_coo(poeDTM[0].tocoo()),300)))
plot_wordcloud(words_only, 'poe', 250)

# Stoker Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(stokerVocab, 
                                               sort_coo(stokerDTM[0].tocoo()),300)))
plot_wordcloud(words_only, 'stoker', 250)

# James Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(jamesVocab, 
                                               sort_coo(jamesDTM[0].tocoo()),300)))
plot_wordcloud(words_only, 'james', 250)

# Book Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(bookVocab, 
                                               sort_coo(bookDTM[0].tocoo()),500)))
plot_wordcloud(words_only, 'all', 450)

# Book Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(predictVocab, 
                                               sort_coo(predictDTM[0].tocoo()),500)))
plot_wordcloud(words_only, 'predict', 450)

# ALL Books Wordclod
# Book Wordcloud
words_only = list(map(itemgetter(0), 
                      extract_topn_from_vector(allbookVocab_STEM, 
                                               sort_coo(allbookDTM_STEM[0].tocoo()),500)))
plot_wordcloud(words_only, 'all', 450)


###------------------------------ Initial Analysis before Data Manipulation -------------------------
# books per author
num_stokerBooks = len(stokerDF)
num_poeBooks = len(poeDF)
num_jamesBooks = len(jamesDF)
num_blackwoodBooks = len(blackwoodDF)

num_book_list =[num_blackwoodBooks, num_poeBooks, num_stokerBooks, num_jamesBooks]
# build a dataframe 
bookFreqDF = []
bookFreqDF = pd.DataFrame(columns = ['book_author', 'num_books'])

index = 0
for auth in author_dict:
    df_row = {'book_author': author_dict.get(auth),
              'num_books': num_book_list[index]}
    # append the row to the bookDF
    bookFreqDF = bookFreqDF.append(df_row, ignore_index = True)
    index = index + 1

# plot with seaborn 
fg = sns.catplot(x = "book_author", y = "num_books", hue = "book_author", dodge=False,
                    height = 3, aspect = 3, palette="Spectral", kind="bar", data=bookFreqDF)
fg.set_xticklabels(rotation=45, horizontalalignment = 'right', 
                   fontweight = 'light', fontsize = 'medium')
fg.set(xlabel = "Author", ylabel = "Number of Works by this Author", 
       title = "Frequency of Works by Each Author")

#-------------------------------- Update Vectorization for Book and Predict -------------------------
# predicted books are 65:end in the index
# remove them from the set
last_index = len(allbookLabel)
predict_index = 65

# separate labelvectorDF
tempDF = allbookLabelVectorDF.copy()
# recreate all the book and predict DFs
bookLabelVectorDF = tempDF.iloc[0:predict_index]
predictLabelVectorDF = tempDF.iloc[predict_index:last_index]

len(bookLabelVectorDF)
len(predictLabel)
len(predictLabelVectorDF)

# separate vectorDF
tempDF = allbookVectorDF.copy()
# recreate all the book and predict DFs
bookVectorDF = tempDF.iloc[0:predict_index]
predictVectorDF = tempDF.iloc[predict_index:last_index]

len(bookVectorDF)
len(predictVectorDF)

# for STEM
# predicted books are 65:end in the index
# remove them from the set
last_index = len(allbookLabel)
predict_index = 65

# separate labelvectorDF
tempDF = allbookLabelVectorDF_STEM.copy()
# recreate all the book and predict DFs
bookLabelVectorDF_STEM = tempDF.iloc[0:predict_index]
predictLabelVectorDF_STEM = tempDF.iloc[predict_index:last_index]

len(bookLabelVectorDF_STEM)
len(predictLabel)
len(predictLabelVectorDF_STEM)

# separate vectorDF
tempDF = allbookVectorDF_STEM.copy()
# recreate all the book and predict DFs
bookVectorDF_STEM = tempDF.iloc[0:predict_index]
predictVectorDF_STEM = tempDF.iloc[predict_index:last_index]

len(bookVectorDF_STEM)
len(predictVectorDF_STEM)

# this will need to be repeated when new vectorization is done

#-------------------------------- Vectorization (binary) for Bernoulli--------------------------------
# for the rest of the vectorization - only doing the full book set (all 4 authors)
# and prediction set
# pass in our own stopwords
allbookCV_b = CountVectorizer(input='content',
                              analyzer = 'word',
                              stop_words = 'english',
                              binary = 'True')

# build total list of just the book text
allbook_text_list = allbookDF['book_text'].tolist()

# execute the transform and then get the feature names (the word - vocabulary)
allbookDTM_b = allbookCV_b.fit_transform(allbook_text_list)

# create the vocabulary list by running feature_names - features being words
allbookVocab_b = allbookCV_b.get_feature_names()

# finally put these in a dataframe - one for each corpus
# this is what is used for test_train
allbookVectorDF_b = pd.DataFrame(allbookDTM_b.toarray(), columns = allbookVocab_b)

# now we need to "attach" the author to each frame
allbookLabel = allbookDF['book_author']
allbookLabelArray = allbookLabel.to_numpy()

# make copies of the final DF, so we can add the label column
allbookLabelVectorDF_b = allbookVectorDF_b.copy()

# insert label into position 0 (first) to *each* dataframe
# this is the *CountVectorizer* vector - BINARY for BernoulliNB
allbookLabelVectorDF_b.insert(0, 'book_author', allbookLabel)

#-------------------------------- Update Vectorization for Book and Predict -------------------------
# predicted books are 65:end in the index
# remove them from the set
# separate labelvectorDF
tempDF = allbookLabelVectorDF_b.copy()
# recreate all the book and predict DFs
bookLabelVectorDF_b = tempDF.iloc[0:predict_index]
predictLabelVectorDF_b = tempDF.iloc[predict_index:last_index]

len(bookLabelVectorDF_b)
len(predictLabelVectorDF_b)

# separate vectorDF
tempDF = allbookVectorDF_b.copy()
# recreate all the book and predict DFs
bookVectorDF_b = tempDF.iloc[0:predict_index]
predictVectorDF_b = tempDF.iloc[predict_index:last_index]

len(bookVectorDF_b)
len(predictVectorDF_b)

#-------------------------------- Vectorization with TFIDF --------------------------------
# for the rest of the vectorization - only doing the full book set (all 4 authors)
# and prediction set
allbookCV_TF = TfidfVectorizer(input='content',
                               analyzer = 'word',
                               stop_words = 'english')

# build total list of just the book text
allbook_text_list = allbookDF['book_text'].tolist()

# execute the transform and then get the feature names (the word - vocabulary)
allbookDTM_TF = allbookCV_TF.fit_transform(allbook_text_list)

# create the vocabulary list by running feature_names - features being words
allbookVocab_TF = allbookCV_TF.get_feature_names()

# finally put these in a dataframe - one for each corpus
# this is what is used for test_train
allbookVectorDF_TF = pd.DataFrame(allbookDTM_TF.toarray(), columns = allbookVocab_TF)

# now we need to "attach" the author to each frame
allbookLabel = allbookDF['book_author']
allbookLabelArray = allbookLabel.to_numpy()

# make copies of the final DF, so we can add the label column
allbookLabelVectorDF_TF = allbookVectorDF_TF.copy()

# insert label into position 0 (first) to *each* dataframe
# this is the *TFIDFVectorizer* vector TFIDF
allbookLabelVectorDF_TF.insert(0, 'book_author', allbookLabel)

#-------------------------------- Update Vectorization for Book and Predict -------------------------
# predicted books are 65:end in the index
# remove them from the set
# separate labelvectorDF
tempDF = allbookLabelVectorDF_TF.copy()
# recreate all the book and predict DFs
bookLabelVectorDF_TF = tempDF.iloc[0:predict_index]
predictLabelVectorDF_TF = tempDF.iloc[predict_index:last_index]

len(bookLabelVectorDF_TF)
len(predictLabelVectorDF_TF)

# separate vectorDF
tempDF = allbookVectorDF_TF.copy()
# recreate all the book and predict DFs
bookVectorDF_TF = tempDF.iloc[0:predict_index]
predictVectorDF_TF = tempDF.iloc[predict_index:last_index]

len(bookVectorDF_TF)
len(predictVectorDF_TF)

#-------------------- Vectorization for ngrams (bi-grams) for MultinomialNB ---------------------
# ** NOTE: this is using non-stemmed data **
# Create an instance of CountVectorizer (one for the corpus)
# this should use both unigrams and bigrams, but I will also do bigrams and trigrams
# now that we have good, clean data
allbookCV_ub = CountVectorizer(input='content',
                               analyzer = 'word',
                               stop_words = 'english',
                               ngram_range = (1,2))

# build total list of just the book text
allbook_text_list = allbookDF['book_text'].tolist()

# execute the transform and then get the feature names (the word - vocabulary)
allbookDTM_ub = allbookCV_ub.fit_transform(allbook_text_list)

# create the vocabulary list by running feature_names - features being words
allbookVocab_ub = allbookCV_ub.get_feature_names()

# finally put these in a dataframe - one for each corpus
# this is what is used for test_train
allbookVectorDF_ub = pd.DataFrame(allbookDTM_ub.toarray(), columns = allbookVocab_ub)

# now we need to "attach" the author to each frame
allbookLabel = allbookDF['book_author']
allbookLabelArray = allbookLabel.to_numpy()

# make copies of the final DF, so we can add the label column
allbookLabelVectorDF_ub = allbookVectorDF_ub.copy()

# insert label into position 0 (first) to *each* dataframe
# this is the *CountVectorizer* vector - no Binary, no TFIDF
allbookLabelVectorDF_ub.insert(0, 'book_author', allbookLabel)

#-------------------------------- Update Vectorization for Book and Predict -------------------------
# predicted books are 65:end in the index
# remove them from the set
# separate labelvectorDF
tempDF = allbookLabelVectorDF_ub.copy()
# recreate all the book and predict DFs
bookLabelVectorDF_ub = tempDF.iloc[0:predict_index]
predictLabelVectorDF_ub = tempDF.iloc[predict_index:last_index]

len(bookLabelVectorDF_ub)
len(predictLabelVectorDF_ub)

# separate vectorDF
tempDF = allbookVectorDF_ub.copy()
# recreate all the book and predict DFs
bookVectorDF_ub = tempDF.iloc[0:predict_index]
predictVectorDF_ub = tempDF.iloc[predict_index:last_index]

len(bookVectorDF_ub)
len(predictVectorDF_ub)

#----------- Vectorization for ngrams (bi-grams and tri-grams) for MultinomialNB ---------------
# ** NOTE: this is using non-stemmed data **
# Create an instance of CountVectorizer (one for the corpus)
# this should use both unigrams and bigrams, but I will also do bigrams and trigrams
# now that we have good, clean data
allbookCV_bt = CountVectorizer(input='content',
                               analyzer = 'word',
                               stop_words = 'english',
                               ngram_range = (2,3))

# build total list of just the book text
allbook_text_list = allbookDF['book_text'].tolist()

# execute the transform and then get the feature names (the word - vocabulary)
allbookDTM_bt = allbookCV_bt.fit_transform(allbook_text_list)

# create the vocabulary list by running feature_names - features being words
allbookVocab_bt = allbookCV_bt.get_feature_names()

# finally put these in a dataframe - one for each corpus
# this is what is used for test_train
allbookVectorDF_bt = pd.DataFrame(allbookDTM_bt.toarray(), columns = allbookVocab_bt)

# now we need to "attach" the author to each frame
allbookLabel = allbookDF['book_author']
allbookLabelArray = allbookLabel.to_numpy()

# make copies of the final DF, so we can add the label column
allbookLabelVectorDF_bt = allbookVectorDF_bt.copy()

# insert label into position 0 (first) to *each* dataframe
# this is the *CountVectorizer* vector - no Binary, no TFIDF
allbookLabelVectorDF_bt.insert(0, 'book_author', allbookLabel)

#-------------------------------- Update Vectorization for Book and Predict -------------------------
# predicted books are 65:end in the index
# remove them from the set
# separate labelvectorDF
tempDF = allbookLabelVectorDF_bt.copy()
# recreate all the book and predict DFs
bookLabelVectorDF_bt = tempDF.iloc[0:predict_index]
predictLabelVectorDF_bt = tempDF.iloc[predict_index:last_index]

len(bookLabelVectorDF_bt)
len(predictLabelVectorDF_bt)

# separate vectorDF
tempDF = allbookVectorDF_bt.copy()
# recreate all the book and predict DFs
bookVectorDF_bt = tempDF.iloc[0:predict_index]
predictVectorDF_bt = tempDF.iloc[predict_index:last_index]

len(bookVectorDF_bt)
len(predictVectorDF_bt)

#------------------------------- Confusion Matrix Pretty Plotter -----------------------
# use some code (used this in our project too)
# to define a function to draw a pretty confusion matrix
# found this at https://scikit-learn.org/0.23/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def my_plot_confusion_matrix(cm, classes,
                             normalize=False,
                             title='Confusion matrix',
                             cmap=plt.cm.viridis_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=3)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.3f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#------------------------------------ Training and Test sets for Modeling -------------------
# create training and testing vars
# for Author with CountVectorizer Vectors
tsize = 0.25
Xbook_train, Xbook_test, ybook_train, ybook_test = train_test_split(bookVectorDF, 
                                                                    bookLabelArray, 
                                                                    test_size=tsize)
## create training and testing vars
# for Author with CountVectorizer Bernulli Vectors
Xbook_train_b, Xbook_test_b, ybook_train_b, ybook_test_b = train_test_split(bookVectorDF_b, 
                                                                            bookLabelArray, 
                                                                            test_size=tsize)
# create training and testing vars
# for Author with TFIDF Vectors
Xbook_train_TF, Xbook_test_TF, ybook_train_TF, ybook_test_TF = train_test_split(bookVectorDF_TF, 
                                                                                bookLabelArray, 
                                                                                test_size=tsize)
# these are only used with MultiNomialNB and not elsewhere
# create training and testing vars
# for Author with CountVectorizer uni,bigrams
Xbook_train_ub, Xbook_test_ub, ybook_train_ub, ybook_test_ub = train_test_split(bookVectorDF_ub, 
                                                                                bookLabelArray, 
                                                                                test_size=tsize)
## create training and testing vars
# for Author with CountVectorizer bi,trigrams Vectors
Xbook_train_bt, Xbook_test_bt, ybook_train_bt, ybook_test_bt = train_test_split(bookVectorDF_bt, 
                                                                                bookLabelArray, 
                                                                                test_size=tsize)

## create training and testing vars
# for Author with CountVectorizer STEMMED Vectors
Xbook_train_STEM, Xbook_test_STEM, ybook_train_STEM, ybook_test_STEM = train_test_split(bookVectorDF_STEM, 
                                                                                        bookLabelArray, 
                                                                                        test_size=tsize)

# no need to set up test/training for predicting sets - they will be used as *test* sets in their 
# entirity - to see if they map to a particular one of the 4 original authors

#------------------------------------------Multinomial Naive Bayes ------------------------------
# the classifiers
# create the Naive Bayes Multinomial classifier (model)
myNB= MultinomialNB()
# Run for different items
bookNB = myNB.fit(Xbook_train, ybook_train)
bookPredict = myNB.predict(Xbook_test)
print(np.round(myNB.predict_proba(Xbook_test),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
bookLabel_CM = confusion_matrix(ybook_test, bookPredict)
print("\nThe confusion matrix is:")
print(bookLabel_CM)

# scores
bookNB.score(Xbook_train, ybook_train)
bookNB.score(Xbook_test, ybook_test)
# Accuracy - test set
AS = accuracy_score(ybook_test, bookPredict)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test, bookPredict, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test, bookPredict, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test, bookPredict, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test, bookPredict, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test, bookPredict, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test, bookPredict, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test, bookPredict))


#--------------------------------- STEMMED Multinomial Naive Bayes ------------------------------
# the classifiers
# create the Naive Bayes Multinomial classifier (model)
myNB_STEM = MultinomialNB()
# Run for different items
bookNB_STEM = myNB_STEM.fit(Xbook_train_STEM, ybook_train_STEM)
bookPredict_STEM = myNB_STEM.predict(Xbook_test_STEM)
print(np.round(myNB_STEM.predict_proba(Xbook_test_STEM),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
bookLabel_CM_STEM = confusion_matrix(ybook_test_STEM, bookPredict_STEM)
print("\nThe confusion matrix is:")
print(bookLabel_CM_STEM)

# scores
bookNB_STEM.score(Xbook_train_STEM, ybook_train_STEM)
bookNB_STEM.score(Xbook_test_STEM, ybook_test_STEM)
# Accuracy - test set
AS = accuracy_score(ybook_test_STEM, bookPredict_STEM)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_STEM, bookPredict_STEM, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_STEM, bookPredict_STEM, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_STEM, bookPredict_STEM, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_STEM, bookPredict_STEM, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_STEM, bookPredict_STEM, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_STEM, bookPredict_STEM, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_STEM, bookPredict_STEM))


#------------------------------------------- Bernoulli Classifier -------------------------------------
# create the Naive Bayes Multinomial classifier (model)
myB_NB= BernoulliNB()
# Run for different items
bookB_NB = myB_NB.fit(Xbook_train_b, ybook_train_b)
bookPredictB = myB_NB.predict(Xbook_test_b)
print(np.round(myB_NB.predict_proba(Xbook_test_b),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
bookLabelB_CM = confusion_matrix(ybook_test_b, bookPredictB)
print("\nThe confusion matrix is:")
print(bookLabelB_CM)

# scores
bookB_NB.score(Xbook_train_b, ybook_train_b)
bookB_NB.score(Xbook_test_b, ybook_test_b)

# Accuracy - test set
AS = accuracy_score(ybook_test_b, bookPredictB)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_b, bookPredictB, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_b, bookPredictB, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_b, bookPredictB, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_b, bookPredictB, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_b, bookPredictB, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_b, bookPredictB, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_b, bookPredictB))

#-------------------------------- TFIDF (with Multinomial NB) Classifier -------------------------------
# create the Naive Bayes Multinomial classifier (model)
myTF_NB= BernoulliNB()
# Run for different items
bookTF_NB = myTF_NB.fit(Xbook_train_TF, ybook_train_TF)
bookPredictTF = myTF_NB.predict(Xbook_test_TF)
print(np.round(myTF_NB.predict_proba(Xbook_test_TF),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
bookLabelTF_CM = confusion_matrix(ybook_test_TF, bookPredictTF)
print("\nThe confusion matrix is:")
print(bookLabelTF_CM)

# scores
bookTF_NB.score(Xbook_train_TF, ybook_train_TF)
bookTF_NB.score(Xbook_test_TF, ybook_test_TF)

# Accuracy - test set
AS = accuracy_score(ybook_test_TF, bookPredictTF)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_TF, bookPredictTF, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_TF, bookPredictTF, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_TF, bookPredictTF, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_TF, bookPredictTF, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookPredictTF, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookPredictTF, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_TF, bookPredictTF))

#----------------------- Multinomial Naive Bayes (Unigrams and Bigrams) ---------------------------
# the classifiers
# create the Naive Bayes Multinomial classifier (model)
myNB = MultinomialNB()
# Run for different items
bookNB_ub = myNB.fit(Xbook_train_ub, ybook_train_ub)
bookPredict_ub = myNB.predict(Xbook_test_ub)
print(np.round(myNB.predict_proba(Xbook_test_ub),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
bookLabel_CM_ub = confusion_matrix(ybook_test_ub, bookPredict_ub)
print("\nThe confusion matrix is:")
print(bookLabel_CM_ub)

# scores
bookNB_ub.score(Xbook_train_ub, ybook_train_ub)
bookNB_ub.score(Xbook_test_ub, ybook_test_ub)

# Accuracy - test set
AS = accuracy_score(ybook_test_ub, bookPredict_ub)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_ub, bookPredict_ub, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_ub, bookPredict_ub, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_ub, bookPredict_ub, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_ub, bookPredict_ub, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_ub, bookPredict_ub, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_ub, bookPredict_ub, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_ub, bookPredict_ub))

#------------------------- Multinomial Naive Bayes (Bigrams and Trigrams) ---------------------------
# the classifiers
# create the Naive Bayes Multinomial classifier (model)
myNB= MultinomialNB()
# Run for different items
bookNB_bt = myNB.fit(Xbook_train_bt, ybook_train_bt)
bookPredict_bt = myNB.predict(Xbook_test_bt)
print(np.round(myNB.predict_proba(Xbook_test_bt),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
bookLabel_CM_bt = confusion_matrix(ybook_test_bt, bookPredict_bt)
print("\nThe confusion matrix is:")
print(bookLabel_CM_bt)

# scores
bookNB_bt.score(Xbook_train_bt, ybook_train_bt)
bookNB_bt.score(Xbook_test_bt, ybook_test_bt)

# Accuracy - test set
AS = accuracy_score(ybook_test_bt, bookPredict_bt)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_bt, bookPredict_bt, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_bt, bookPredict_bt, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_bt, bookPredict_bt, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_bt, bookPredict_bt, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_bt, bookPredict_bt, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_bt, bookPredict_bt, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_bt, bookPredict_bt))

#----------------------- Support Vector Machines ---------------------------------------------
#---- Now do the same with SVM and 3 different kernels - with two different vectorizers ------
#------------------------------------ SVM with Linear kernel ---------------------------------
# with Countvectorizer
SVM_Model=LinearSVC(C=1)
bookSVM_Linear = SVM_Model.fit(Xbook_train, ybook_train)
bookSVMPredict = SVM_Model.predict(Xbook_test)

SVM_matrix = confusion_matrix(ybook_test, bookSVMPredict)
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")

# scores
bookSVM_Linear.score(Xbook_train, ybook_train)
bookSVM_Linear.score(Xbook_test, ybook_test)

# Accuracy - test set
AS = accuracy_score(ybook_test, bookSVMPredict)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test, bookSVMPredict, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test, bookSVMPredict, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test, bookSVMPredict, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test, bookSVMPredict))

#-------------------------------------------------------------------------
# with TFIDF
bookSVM_LinearTF = SVM_Model.fit(Xbook_train_TF, ybook_train_TF)
bookSVMPredictTF = SVM_Model.predict(Xbook_test_TF)

SVM_matrixTF = confusion_matrix(ybook_test_TF, bookSVMPredictTF)
print("\nThe confusion matrix is:")
print(SVM_matrixTF)
print("\n\n")

# scores
bookSVM_LinearTF.score(Xbook_train_TF, ybook_train_TF)
bookSVM_LinearTF.score(Xbook_test_TF, ybook_test_TF)

# Accuracy - test set
AS = accuracy_score(ybook_test_TF, bookSVMPredictTF)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_TF, bookSVMPredictTF, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_TF, bookSVMPredictTF, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_TF, bookSVMPredictTF, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_TF, bookSVMPredictTF))

#------------------------------------ SVM with rbf kernel ----------------------------------------
# ran with Cost = 1 - all classified as one label
# ran again with Cost = 50, much better
SVM_Model_rbf=svm.SVC(C=100, kernel='rbf', 
                      verbose=True, gamma="auto")
# with Countvectorizer
bookSVM_rbf = SVM_Model_rbf.fit(Xbook_train, ybook_train)
bookSVMPredict_rbf = SVM_Model_rbf.predict(Xbook_test)

SVM_matrix_rbf = confusion_matrix(ybook_test, bookSVMPredict_rbf)
print("\nThe confusion matrix is:")
print(SVM_matrix_rbf)
print("\n\n")

# scores
bookSVM_rbf.score(Xbook_train, ybook_train)
bookSVM_rbf.score(Xbook_test, ybook_test)

# Accuracy - test set
AS = accuracy_score(ybook_test, bookSVMPredict_rbf)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test, bookSVMPredict_rbf, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test, bookSVMPredict_rbf, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test, bookSVMPredict_rbf, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict_rbf, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict_rbf, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict_rbf, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test, bookSVMPredict_rbf))

#-------------------------------- SVM with RBF Kernel TFIDF Vectorization -----------
# with TFIDF

bookSVM_TF_rbf = SVM_Model_rbf.fit(Xbook_train_TF, ybook_train_TF)
bookSVMPredictTF_rbf = SVM_Model_rbf.predict(Xbook_test_TF)

SVM_matrixTF_rbf = confusion_matrix(ybook_test_TF, bookSVMPredictTF_rbf)
print("\nThe confusion matrix is:")
print(SVM_matrixTF_rbf)
print("\n\n")

# scores
bookSVM_TF_rbf.score(Xbook_train_TF, ybook_train_TF)
bookSVM_TF_rbf.score(Xbook_test_TF, ybook_test_TF)

# Accuracy - test set
AS = accuracy_score(ybook_test_TF, bookSVMPredictTF_rbf)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_TF, bookSVMPredictTF_rbf, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_TF, bookSVMPredictTF_rbf, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_rbf, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_rbf, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_rbf, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_rbf, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_TF, bookSVMPredictTF_rbf))

#------------------------------------ SVM with poly kernel ----------------------------------------
SVM_Model_poly=svm.SVC(C=100, kernel='poly', degree=2,
                       verbose=True, gamma="auto")

# with Countvectorizer
bookSVM_poly = SVM_Model_poly.fit(Xbook_train, ybook_train)
bookSVMPredict_poly = SVM_Model_poly.predict(Xbook_test)

SVM_matrix_poly = confusion_matrix(ybook_test, bookSVMPredict_poly)
print("\nThe confusion matrix is:")
print(SVM_matrix_poly)
print("\n\n")

# scores
bookSVM_poly.score(Xbook_train, ybook_train)
bookSVM_poly.score(Xbook_test, ybook_test)

# Accuracy - test set
AS = accuracy_score(ybook_test, bookSVMPredict_poly)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test, bookSVMPredict_poly, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test, bookSVMPredict_poly, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test, bookSVMPredict_poly, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict_poly, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict_poly, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test, bookSVMPredict_poly, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test, bookSVMPredict_poly))

#----------------------------------------------
# with TFIDF
bookSVM_TF_poly = SVM_Model_poly.fit(Xbook_train_TF, ybook_train_TF)
bookSVMPredictTF_poly = SVM_Model_poly.predict(Xbook_test_TF)

SVM_matrixTF_poly = confusion_matrix(ybook_test_TF, bookSVMPredictTF_poly)
print("\nThe confusion matrix is:")
print(SVM_matrixTF_poly)
print("\n\n")

# scores
bookSVM_TF_poly.score(Xbook_train_TF, ybook_train_TF)
bookSVM_TF_poly.score(Xbook_test_TF, ybook_test_TF)

# Accuracy - test set
AS = accuracy_score(ybook_test_TF, bookSVMPredictTF_poly)
print("Accuracy is ", AS)
# Recall
RS = recall_score(ybook_test_TF, bookSVMPredictTF_poly, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(ybook_test_TF, bookSVMPredictTF_poly, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_poly, average=None)
print("F1 Score is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_poly, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_poly, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(ybook_test_TF, bookSVMPredictTF_poly, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(ybook_test_TF, bookSVMPredictTF_poly))

#------------------------------- Confusion Matrix Pretty Plotter -----------------------
# use some code (used this in our project too)
# to define a function to draw a pretty confusion matrix
# found this at https://scikit-learn.org/0.23/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def my_plot_confusion_matrix(cm, classes,
                             normalize=False,
                             title='Confusion matrix',
                             cmap=plt.cm.viridis_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=3)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.3f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Author')
    plt.xlabel('Predicted Author')

#-------------------------------------- Plotting ------------------------------------------------
# Multinomial NB with CountVectorizer
# Plot non-normalized confusion matrix
lbls = ['Blackwood', 'James', 'Poe', 'Stoker']
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabel_CM, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - Multinomial Naive Bayes\nVectorization with Count Vectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabel_CM, classes=lbls, normalize = True,
                         title='Book Labeled by Author Normalized Confusion Matrix  - Multinomial Naive Bayes\nVectorization with Count Vectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Multinomial NB with TFIDF Vectorizer
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabelTF_CM, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - Multinomial Naive Bayes\nVectorization with TFIDF Vectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabelTF_CM, classes=lbls, normalize = True,
                         title='Book Labeled by Author Normalized Confusion Matrix - Multinomial Naive Bayes\nVectorization with TFIDF Vectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Bernoulli NB with CountVectorizer
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabelB_CM, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - Bernoulli Naive Bayes\nVectorization with Binary Count Vectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabelB_CM, classes=lbls, normalize = True,
                         title='Book Labeled by Author Normalized Confusion Matrix - Bernoulli Naive Bayes\nVectorization with Binary Count Vectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Multinomial NB - Unigrams and Bigrams
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabel_CM_ub, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - Multinomial Naive Bayes\nVectorization with Multinomial Vectorizer\nUnigrams and Bigrams',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabel_CM_ub, classes=lbls, normalize = True,
                         title='Book Labeled by Author Normalized Confusion Matrix - Multinomial Naive Bayes\nVectorization with Multinomial Vectorizer\nUnigrams and Bigrams',
                         cmap=plt.cm.Reds)
plt.show()

# Multinomial NB - Bigrams and Trigrams 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabel_CM_bt, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - Multinomial Naive Bayes\nVectorization with Binary Count Vectorizer\nBigrams and Trigrams',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(bookLabel_CM_bt, classes=lbls, normalize = True,
                         title='Book Labeled by Author Normalized Confusion Matrix - Multinomial Naive Bayes\nVectorization with Binary Count Vectorizer\nBigrams and Trigrams',
                         cmap=plt.cm.Reds)
plt.show()

# SVM Linear Kernel with CountVectorizer Vectorization 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrix, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - SVM with Linear Kernel\nVectorization with CountVectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrix, classes=lbls, normalize = True,
                         title='Book Labeled by Author Confusion Matrix - SVM with Linear Kernel\nVectorization with CountVectorizer (Normalized)',
                         cmap=plt.cm.Reds)
plt.show()

# SVM Linear Kernel with TFIDFVectorizer Vectorization 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrixTF, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - SVM with Linear Kernel\nVectorization with TFIDFVectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrixTF, classes=lbls, normalize = True,
                         title='Book Labeled by Author Confusion Matrix - SVM with Linear Kernel\nVectorization with TFIDFVectorizer (Normalized)',
                         cmap=plt.cm.Reds)
plt.show()

# SVM RBF Kernel with CountVectorizer Vectorization 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrix_rbf, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - SVM with RBF Kernel\nVectorization with CountVectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrix_rbf, classes=lbls, normalize = True,
                         title='Book Labeled by Author Confusion Matrix - SVM with RBF Kernel\nVectorization with CountVectorizer (Normalized)',
                         cmap=plt.cm.Reds)
plt.show()

# SVM RBF with TFIDFVectorizer Vectorization 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrixTF_rbf, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - SVM with RBF Kernel\nVectorization with TFIDFVectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrixTF_rbf, classes=lbls, normalize = True,
                         title='Book Labeled by Author Confusion Matrix - SVM with RBF Kernel\nVectorization with TFIDFVectorizer (Normalized)',
                         cmap=plt.cm.Reds)
plt.show()

# SVM Polynomial Kernel with CountVectorizer Vectorization 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrix_poly, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - SVM with Polynomial Kernel\nVectorization with CountVectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrix_poly, classes=lbls, normalize = True,
                         title='Book Labeled by Author Confusion Matrix - SVM with Polynomial Kernel\nVectorization with CountVectorizer (Normalized)',
                         cmap=plt.cm.Reds)
plt.show()


# SVM Polynomial Kernel with TFIDFVectorizer Vectorization 
#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrixTF_poly, classes=lbls, normalize = False,
                         title='Book Labeled by Author Confusion Matrix - SVM with Polynomial Kernel\nVectorization with TFIDFVectorizer',
                         cmap=plt.cm.Reds)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(6,4))
my_plot_confusion_matrix(SVM_matrixTF_poly, classes=lbls, normalize = True,
                         title='Book Labeled by Author Confusion Matrix - SVM with Polynomial Kernel\nVectorization with TFIDFVectorizer (Normalized)',
                         cmap=plt.cm.Reds)
plt.show()

#---------------------------------- Cross Validation --------------------------------------------
# The function works for Bernoulli, MultinomalNB, SVM - Linear, Poly and RBF Kernels
# it also supports the passing of a cost variable for SVM 
# items passed - then we can use this information to pass in test data from the predicted set
###------------------------------- Function for doing Cross Validation ---------------------------
# Author: These programs were provided to me in IST 664 Class by the Instructor
# first define the function to run the cross validation given the docs, gold standard labels and the 
# the featureset for testing
# function passing number of folds, feature set and labels
## cross-validation ##
# calling cross_validation_accuracy(num_folds, restSentDF, 'sentiment')
###------------------------------- Function for doing Cross Validation ---------------------------
# Author: These programs were provided to me in IST 664 Class by the Instructor
# first define the function to run the cross validation given the docs, gold standard labels and the 
# the featureset for testing
# function passing number of folds, feature set and labels
## cross-validation ##
# calling cross_validation_accuracy(num_folds, restSentDF, 'sentiment')
def cross_validation_accuracy(num_folds, df_wlabels, lbl_col, 
                              modelName = 'MultiNB', cost_value=1):
    subset_size = int(len(df_wlabels)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        #print("Fold is ", i)
        test_this_round = df_wlabels[(i*subset_size):][:subset_size]
        #print ("test round is ", test_this_round.head())
        # build the training set
        if i != 0:
            firstHalf = df_wlabels[:(i*subset_size)]
            secondHalf = df_wlabels[((i+1)*subset_size):]
            train_this_round = firstHalf.append(secondHalf)
        else:
            train_this_round = df_wlabels[((i+1)*subset_size):]
        #print ("train round is ", train_this_round.head())
        # build train with and without labels
        X_train = train_this_round.loc[:, train_this_round.columns != lbl_col]
        y_train = train_this_round[lbl_col]
        #print("X_train head is ", X_train.head())
        #print("y_train head is ", y_train.head())
        # build test with and without labels
        #X_test = train_this_round.loc[:, train_this_round.columns != lbl_col]
        #y_test = train_this_round[lbl_col]
        X_test = test_this_round.loc[:, test_this_round.columns != lbl_col]
        y_test = test_this_round[lbl_col]
        #print("X_test head is ", X_test.head())
        #print("y_test head is ", y_test.head())
        # train using train_this_round
        # create classifer based on modelName
        if modelName == 'MultiNB' or modelName == 'TFIDF':
            # create the Naive Bayes Multinomial classifier (model)
            myModel = MultinomialNB()
        if modelName == 'Bernoulli':
            myModel = BernoulliNB()
        # cost value = 1 is good here
        if modelName == 'Linear':
            myModel = LinearSVC(C = cost_value)
        # cost value = 100 is good here
        if modelName == 'RBF':
            myModel = svm.SVC(C = cost_value, kernel = 'rbf', 
                              verbose = True, gamma = "auto")
        # cost value = 100 is good here
        if modelName == 'Poly':
            myModel = svm.SVC(C = cost_value, kernel = 'poly', 
                              degree = 2, verbose = True, 
                              gamma = "auto")
        fitModel = myModel.fit(X_train, y_train)
        predictModel = myModel.predict(X_test)
        y_predict = predictModel
        # evaluate against test_this_round and save accuracy
        # Confusion Matrix
        # call confusion matrix with TEST Labels, PREDICTED Labels
        CM = confusion_matrix(y_test, y_predict)
        # Accuracy
        AS = accuracy_score(y_test, y_predict)
        print("Fold: ", i , "Accuracy Score: ", AS)
        # Recall
        RS = recall_score(y_test, y_predict, average=None)
        # Precision
        PS = precision_score(y_test, y_predict, average=None)

        # Method 3: Classification report [BONUS]
        #print(classification_report(y_test, y_predict))
        # add accuracy to list
        accuracy_list.append(AS)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

# test this cross-validation function
# need to do a random shuffle
# might need them combined before shuffling, and then split out the sentiment column
# set the number of folds for cross validation
num_folds = 10
# CountVectorizer - with Multinomial NB (Book Author)
print("\nMultinomial Naive Bayes with CountVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF = shuffle(bookLabelVectorDF)
cross_validation_accuracy(num_folds, s_bookLabelVectorDF, 'book_author', 'MultiNB')

# TFIDFVectorizer - with Multinomial NB (Book Author)
print("\nMultinomial Naive Bayes with TFIDFVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF_TF = shuffle(bookLabelVectorDF_TF)
cross_validation_accuracy(num_folds, s_bookLabelVectorDF_TF, 'book_author', 'TFIDF')

# CountVectorizer - with Bernoulli NB (Book Author)
print("\nBernoulli Naive Bayes with CountVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF = shuffle(bookLabelVectorDF)
cross_validation_accuracy(num_folds, s_bookLabelVectorDF, 'book_author', 'Bernoulli')

# CountVectorizer - with SVM - Linear Kernel (Book Author)
print("\nSVM Linear Kernel with CountVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF = shuffle(bookLabelVectorDF)
# passing in Cost_value = 1
cross_validation_accuracy(num_folds, s_bookLabelVectorDF, 'book_author', 'Linear', 1)

# TFIDFVectorizer - with SVM - Linear Kernel (Book Author)
print("\nSVM Linear Kernel with TFIDFVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF_TF = shuffle(bookLabelVectorDF_TF)
# passing in Cost_value = 1
cross_validation_accuracy(num_folds, s_bookLabelVectorDF_TF, 'book_author', 'Linear', 1)

# CountVectorizer - with SVM RBF Kernel (Book Author)
print("\nSVM RBF Kernel with CountVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF = shuffle(bookLabelVectorDF)
# passing in Cost_value = 100
cross_validation_accuracy(num_folds, s_bookLabelVectorDF, 'book_author', 'RBF', 100)

# TFIDFVectorizer - with SVM RBF Kernel (Book Author)
print("\nSVM RBF Kernel with TFIDFVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF_TF = shuffle(bookLabelVectorDF_TF)
# passing in Cost_value = 100
cross_validation_accuracy(num_folds, s_bookLabelVectorDF_TF, 'book_author', 'RBF', 100)

# CountVectorizer - with SVM Poy Kernel (Book Author)
print("\nSVM Polynomial Kernel with CountVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF = shuffle(bookLabelVectorDF)
# passing in Cost_value = 100
cross_validation_accuracy(num_folds, s_bookLabelVectorDF, 'book_author', 'Poly', 100)

# TFIDFVectorizer - with SVM Polynomial Kernel (Book Author)
print("\nSVM Polynomial Kernel with TFIDFVectorizer Results (Book Labeled by Author):")
s_bookLabelVectorDF_TF = shuffle(bookLabelVectorDF_TF)
# passing in Cost_value = 100
cross_validation_accuracy(num_folds, s_bookLabelVectorDF_TF, 'book_author', 'Poly', 100)

#-------------------------- Testing with Predict ----------------------------------
### OK - time to run the *BEST* models on our predicted set
# Best Models:
# Multinomial NB using CountVectorizer vectorization
# first just use one test using 
# predictVectorDF - is the DF that has the words and vectors as a dataframe (like Xbook_test)
# PredictLabelArray - is the array of correct labels in an array (like ybook_test)

# let's create just a test set of the authors from our set of four
four_predictLabelVectorDF = predictLabelVectorDF[predictLabelVectorDF['book_author'].isin(['poe','stoker', 'james', 'blackwood'])]
four_predictLabel = four_predictLabelVectorDF['book_author']
four_predictLabelArray = four_predictLabel.to_numpy()
four_predictDF = four_predictLabelVectorDF.drop(['book_author'], axis=1)
#------------------------------------------Multinomial Naive Bayes ------------------------------
# the classifiers
# create the Naive Bayes Multinomial classifier (model)
myNB= MultinomialNB()
bookNB = myNB.fit(Xbook_train, ybook_train)
# this doesn't work....
predictPredict = myNB.predict(four_predictDF)
print(np.round(myNB.predict_proba(predictVectorDF),2))

# call confusion matrix with TEST Labels, PREDICTED Labels
predictLabel_CM = confusion_matrix(four_predictLabelArray, predictPredict)
print("\nThe confusion matrix is:")
print(predictLabel_CM)

# Accuracy - test set
AS = accuracy_score(four_predictLabelArray, predictPredict)
print("Accuracy is ", AS)
# Recall
RS = recall_score(four_predictLabelArray, predictPredict, average=None)
print("Recall is ", RS)
# Precision
PS = precision_score(four_predictLabelArray, predictPredict, average=None)
print("Precision Score is ", PS)

# Method 1: sklearn
F1 = f1_score(four_predictLabelArray, predictPredict, average=None)
print("F1 Score is ", F1)
F1 = f1_score(four_predictLabelArray, predictPredict, average='micro')
print("F1 Micro Average is ", F1)
F1 = f1_score(four_predictLabelArray, predictPredict, average='macro')
print("F1 Macro Average is ", F1)
F1 = f1_score(four_predictLabelArray, predictPredict, average='weighted')
print("F1 Weighted Average is ", F1)

# Method 3: Classification report [BONUS]
print(classification_report(four_predictLabelArray, predictPredict))

#------------------------------------------------------------------
# New function for confusion matrix
def my_plot_confusion_matrix_uneven(cm, xclasses, yclasses,
                                    normalize=False,
                                    title='Confusion Matrix',
                                    cmap=plt.cm.viridis_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=3)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    xtick_marks = np.arange(len(xclasses))
    ytick_marks = np.arange(len(yclasses))
    plt.xticks(xtick_marks, xclasses, rotation=0)
    plt.yticks(ytick_marks, yclasses)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.3f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Author')
    plt.xlabel('Predicted Author')

#----------------------------------------- Now do for entire set ----------------------------------------
# Multinomial Naive Bayes with CountVectorizer Vectorization ------------------------------
# the classifiers
# create the Naive Bayes Multinomial classifier (model)
# This is the instance of the model - it does not change
myNB= MultinomialNB()
bookNB = myNB.fit(Xbook_train, ybook_train)
# this doesn't work....
storyPredict = myNB.predict(predictVectorDF)
myNB.predict_proba(predictVectorDF)
story_CM = np.round(myNB.predict_proba(predictVectorDF),2)

#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plbls = predictLabelArray.tolist()
plt.figure(figsize=(20,20))
# Plot normalized confusion matrix
my_plot_confusion_matrix_uneven(story_CM, xclasses=lbls, yclasses=plbls, normalize = True,
                                title='Predicted Author (unseen) Confusion Matrix - MultinomialNB\nVectorization with CountVectorizer (Normalized)',
                                cmap=plt.cm.Reds)
plt.show()

# Bernoulli NB CountVectorizer vectorization ------------------------
myB =BernoulliNB()
bookB = myB.fit(Xbook_train, ybook_train)
storyPredictB = myB.predict(predictVectorDF)

myB.predict_proba(predictVectorDF)
story_CMB = np.round(myB.predict_proba(predictVectorDF),2)

#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
plbls = predictLabelArray.tolist()
plt.figure(figsize=(20,20))
# Plot normalized confusion matrix
my_plot_confusion_matrix_uneven(story_CMB, xclasses=lbls, yclasses=plbls, normalize = True,
                                title='Predicted Author (unseen) Confusion Matrix - BernoulliNB\nVectorization with CountVectorizer (Normalized)',
                                cmap=plt.cm.Reds)
plt.show()

## 3/12 - ** WORKS TO HERE **
# SVM Linear Kernel, Cost = 1 CountVectorizer vectorization ------------------------
#SVM_Model=LinearSVC(C=1)
#bookSVM_Linear = SVM_Model.fit(Xbook_train, ybook_train)
#storySVMPredict = SVM_Model.predict(predictVectorDF)

# need to find another way to get data here ** JOYCE **
#SVM_Model.predict_proba(predictVectorDF)
#SVMLinear_CM = np.round(SVM_Model.predict_proba(predictVectorDF),2)

#-------------------------------------- Plotting ------------------------------------------------
# Plot non-normalized confusion matrix
#plbls = predictLabelArray.tolist()
#plt.figure(figsize=(20,20))
# Plot normalized confusion matrix
#my_plot_confusion_matrix_uneven(SVMLinear_CM, xclasses=lbls, yclasses=plbls, normalize = True,
#                                title='Predicted Author (unseen) Confusion Matrix - Linear SVM\nVectorization with CountVectorizer (Normalized)',
#                                cmap=plt.cm.Reds)
#plt.show()

#### --------------------------------- LDA for Topic Modeling ---------------------------

## implement a print function
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
        
# Build the topic model using 6 topics
lda_model = LatentDirichletAllocation(n_components=6, max_iter=10, learning_method='online')
LDA_Model = lda_model.fit_transform(allbookVectorDF)

print("SIZE: ", LDA_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
print("First Book in Gothis Horror Book Corpus...")
print(LDA_Model[0])
print("Seventh Book in Gotic Horror Book Corpus...")
print(LDA_Model[6])

## Print LDA using print function from above
print("LDA Horse Book Model:")
print_topics(lda_model, allbookCV)

# print top 10 words for each topic
for i,topic in enumerate(lda_model.components_):
    print(f'Top 10 words for topic #{i}:')
    print([allbookCV.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = lda_model.transform(allbookVectorDF)
topic_values.shape

#------------------------------No stemming----------------------------------
# topic matrix
# Create Document — Topic Matrix
#lda_output = best_lda_model.transform(data_vectorized)# column names
lda_output = lda_model.transform(allbookVectorDF)
#topicnames = [“Topic” + str(i) for i in range(best_lda_model.n_components)]# index names
topicnames = ['Topic' + str(i) for i in range(lda_model.n_components)]
# this needs to be the booknames!!!!
#docnames = ['Doc' + str(i) for i in range(len(data))]# Make the pandas dataframe
docnames = allbookLabel

booknames = []
book_list_array = allbookDF['book_name']

for book_name in book_list_array:
    #print(bname)
    bname = book_dict.get(book_name)
    booknames = booknames + [bname]
len(booknames)

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)# Get dominant topic for each document

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic# Styling
df_document_topic['Book Name'] = booknames
len(df_document_topic)
df_document_topic.head(94)

authnames = []
for auth_name in allbookLabel:
    #print(bname)
    aname = predict_author_dict.get(auth_name)
    authnames = authnames + [aname]

## JOYCE GENERATE WITHOUT FANCY COLORING!
fancyDF = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
fancyDF['dominant_topic'] = dominant_topic
fancyDF['Author Name'] = authnames  
fancyDF['Book Name'] = booknames 
fancyDF = pd.DataFrame(fancyDF.set_index('Author Name'))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    # more options can be specified also
    print(fancyDF)


def color_green(val):
 color = 'green' if val > .1 else 'black'
 return 'color: {col}'.format(col=color)

def make_bold(val):
 weight = 700 if val > .1 else 400
 return 'font-weight: {weight}'.format(weight=weight)# Apply Style

#df_document_topics = df_document_topic.head(36).style.applymap(color_green).applymap(make_bold)
#df_document_topics
#display(df_document_topics)

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model.components_)# Assign Column and Index
df_topic_keywords.columns = allbookCV.get_feature_names()
df_topic_keywords.index = topicnames# View
df_topic_keywords.head()


# Show top n keywords for each topic
def show_topics(vectorizer=allbookCV, lda_model=lda_model, n_words=20):
    keywords = np.array(allbookCV.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=allbookCV, lda_model=lda_model, n_words=15)# Topic - Keywords Dataframe

df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords

# so can see the entire dataframe
pd.set_option('display.max_rows', 1000)

# for plotting
fancyDF_melted = pd.melt(fancyDF, id_vars=["dominant_topic"], ignore_index=False, 
                         value_vars=["Topic0", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5"])
fancyDF_melted.head(36)
len(fancyDF_melted)
# remove all 0' values
fancyDF_melted = fancyDF_melted[(fancyDF_melted[['value']] != 0.00).all(axis=1)]
len(fancyDF_melted)
display(fancyDF_melted)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    # more options can be specified also
    print(fancyDF_melted)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize']=16,8

ax = sns.scatterplot(data = fancyDF_melted, x = fancyDF_melted.index, y="value", 
                     hue = 'variable', palette = 'Set1', s=100)
                     #size = 'value')

plt.xticks(rotation=90, horizontalalignment = 'center', fontweight = 'light', fontsize = 'medium')
plt.title("Dominant Topics for All Gothic Horror Books", fontsize = 12)
plt.legend(shadow=True, fancybox=True,loc='right', bbox_to_anchor = (1.1, 0.5)) 
plt.xlabel("Author Name", fontsize = 10)
plt.ylabel("Probability for Topic", fontsize = 10)

###------------------------------ How many Books per Topic -------------------------
## Which tracks have the most infractions?
# count infractions by Race Track
topicsDFbyBook = fancyDF_melted['dominant_topic'].value_counts()
# convert to dataframe
topicsDFbyBook = topicsDFbyBook.to_frame()
# index by Track Name
dominantNames = topicsDFbyBook.index.values
# rename column to "Infractions"
topicsDFbyBook.columns = ['Books in Topic']

topicsDFbyBook['Topic Number'] = dominantNames

# plot with seaborn
fg = sns.catplot(x = "Topic Number", y = "Books in Topic", hue = "Topic Number", dodge=False,
                    height = 5, aspect = 2, palette="Spectral", kind="bar", data=topicsDFbyBook)
fg.set_xticklabels(horizontalalignment = 'right', 
                         fontweight = 'light', fontsize = 'medium')
fg.set(xlabel = "Topic Number", ylabel = "Number of Books in Topic", 
       title = "Gothic Horror Authors by Topic - LDA Six Topics, no Stemming")



####################################################
##
## VISUALIZATION
##
####################################################

import pyLDAvis.sklearn as LDAvis
import pyLDAvis

## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
# MyVectLDA_DH -> MyTxtCV
# Vect_DH -> MyTxtDTM
# CorpusDF_DH -> MyTxtDF

#panel = LDAvis.prepare(lda_model_DH, Vect_DH, MyVectLDA_DH, mds='tsne')
panel = LDAvis.prepare(lda_model, allbookDTM, allbookCV, mds='tsne')

### !!!!!!! Important - you must interrupt and close the kernet in Spyder to end
## In other words - press the red square and then close the small red x to close
## the Console
pyLDAvis.show(panel)




