import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from string import punctuation
import re
import random
import collections
from nltk.util import ngrams
import copy
import nltk
from collections import Counter
nltk.download('punkt')
from itertools import permutations
import decimal

with open('/Users/ryanklapper/Desktop/2020/Data Science/SongGen/main.txt') as f:
    vocab = {}
    word_count = {}
    words = [word for line in f for word in line.split()]
    #now use collections.Counter
    c = Counter(words)

def remove_punctuation(doc):
    for word in punctuation_symbols:
        doc = doc.replace(*word)
    return doc


def fix_contraction(doc):
    doc = re.sub(r"ain't", "are not", doc)
    doc = re.sub(r"could've", "could have", doc)
    doc = re.sub(r"couldn't", "could not", doc)
    doc = re.sub(r"can't", "can not", doc)
    doc = re.sub(r"won't", "will not", doc)
    doc = re.sub(r"don't", "do not", doc)
    doc = re.sub(r"didn't", "did not", doc)
    doc = re.sub(r"doesn't", "does not", doc)
    doc = re.sub(r"hadn't", "had not", doc)
    doc = re.sub(r"haven't", "have not", doc)
    doc = re.sub(r"hasn't", "has not", doc)
    doc = re.sub(r"he'd", "he would", doc)
    doc = re.sub(r"he'll", "he will", doc)
    doc = re.sub(r"he's", "he is", doc)
    doc = re.sub(r"how'd", "how did", doc)
    doc = re.sub(r"how'll", "how will", doc)
    doc = re.sub(r"how's", "how is", doc)
    doc = re.sub(r"i'd", "i would", doc)
    doc = re.sub(r"i'll", "i will", doc)
    doc = re.sub(r"i'm", "i am", doc)
    doc = re.sub(r"i've", "i have", doc)
    doc = re.sub(r"isn't", "is not", doc)
    doc = re.sub(r"it's", "it is", doc)
    doc = re.sub(r"it'd", "it would", doc)
    doc = re.sub(r"it'd've", "it would have", doc)
    doc = re.sub(r"let's", "let us", doc)
    doc = re.sub(r"might've", "might have", doc)
    doc = re.sub(r"must've", "must have", doc)
    doc = re.sub(r"she'd", "she would", doc)
    doc = re.sub(r"she'll", "she will", doc)
    doc = re.sub(r"she's", "she has", doc)
    doc = re.sub(r"should've", "should have", doc)
    doc = re.sub(r"shouldn't", "should not", doc)
    doc = re.sub(r"that'd", "that would", doc)
    doc = re.sub(r"that's", "that is", doc)
    doc = re.sub(r"there's", "there is", doc)
    doc = re.sub(r"they'd", "they had", doc)
    doc = re.sub(r"they'll", "they will", doc)
    doc = re.sub(r"they've", "they have", doc)
    doc = re.sub(r"they're", "they are", doc)
    doc = re.sub(r"wasn't", "was not", doc)
    doc = re.sub(r"we'd", "we had", doc)
    doc = re.sub(r"we've", "we have", doc)
    doc = re.sub(r"we'll", "we will", doc)
    doc = re.sub(r"we're", "we are", doc)
    doc = re.sub(r"weren't", "were not", doc)
    doc = re.sub(r"what's", "what is", doc)
    doc = re.sub(r"would've", "would have", doc)
    doc = re.sub(r"wouldn't", "would not", doc)
    doc = re.sub(r"where'd", "where did", doc)
    doc = re.sub(r"where's", "where is", doc)
    doc = re.sub(r"who's", "who is", doc)
    doc = re.sub(r"who'd", "who would", doc)
    doc = re.sub(r"why'd", "why did", doc)
    doc = re.sub(r"y'all", "you all", doc)
    doc = re.sub(r"you'd", "you would", doc)
    doc = re.sub(r"you're", "you are", doc)
    doc = re.sub(r"you've", "you have", doc)
    doc = re.sub(r"you'll", "you will", doc)

    return doc


punctuation_symbols = []
for each in list(punctuation):
    punctuation_symbols.append((each, ''))


    #cleaning

for idx in range(len(words)):
    words[idx] = words[idx].lower()
    words[idx] = fix_contraction(words[idx])
    words[idx] = re.sub(r'\s+', ' ', words[idx])  # remove newline chars
    words[idx] = re.sub(r"\'", "", words[idx])  # remove single quotes
    words[idx] = re.sub(r'\S*_\S*\s?', '', words[idx]) #removes underscores
    words[idx] = re.sub(r'\S*-\S*\s?', '', words[idx]) #removes dashes
    words[idx] = remove_punctuation(words[idx]) #removes all of the other symbols


words1 = [word for line in words for word in line.split()]

#first get word counts using nltk package
fdist = FreqDist(words1)
fdist
#convert into dictionary
wordDict = dict(fdist)


dict1 = {}
counter = 0
# Index Dictionary Mapping
for key, value in wordDict.items():
    dict1[counter] = key
    counter += 1
# Converting words1 and tokenizing it for bigram function NLTK package usage
wordsNew = copy.deepcopy(words1)
w3 = ' '.join(wordsNew)
token = nltk.word_tokenize(w3)
print(token)
# Creating bigrams
bigrams = ngrams(token,2)
s = copy.deepcopy(Counter(bigrams))
# Creating bigram dictionaries for probabiltiy calculations
bigramDict = dict(s)
bigramDict1 = copy.deepcopy(bigramDict)
bigramDict2 = copy.deepcopy(bigramDict1)
# Creating probability dictionary
probIndex = {}
for i, g in bigramDict1.items():
    firstWord = i[0]
    totalFreq = 0
    totalsum = 0
    for k, v in bigramDict1.items():
        if(k[0] == firstWord):
            totalFreq+= v
    for k, v in bigramDict1.items():
        if(k[0] == firstWord):
            bigramProb = (v / totalFreq)
            totalsum+= bigramProb
            probIndex[k] = bigramProb

print(probIndex)
#create random value from 0 to 1
#select a random word from original list (unigram)
#find all the bigrams that start with that unigram
#do probability range and select a word
randomNum = decimal.Decimal(random.randrange(0,100))/100
print(randomNum)
#random index
#randomIndex = random.randint(0,len(dict1))
#grab word given index
#randomWord = dict1[randomIndex]
#random index
randomIndex = random.randint(0,len(dict1))

#grab word given index
randomWord = dict1[randomIndex]
probSum = 0
listofWords = []
# count = 0

s = " "
master_song = ''
master_song += randomWord
master_song += " "
n = 0
length_of_song = 150

for i in range(0, length_of_song - 1 ):
    for k,v in probIndex.items():
#print(count, k[0])
#if random word is equal to the first word in a bigram then
        if(randomWord == k[0]):
            probSum += float(v)

            if(randomNum < probSum):
                s += k[1]
                s += " "
                probSum1 = -1
            else:
                print('---')

    tokenL = nltk.word_tokenize(s)
    master_song += str(tokenL[n])
    master_song += " "

    randomWord = str(tokenL[n])
    n += 1

print(master_song)
