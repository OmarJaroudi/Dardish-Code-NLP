# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:23:42 2019

@author: Omar Al Jaroudi
"""
#libararies
import pandas as pd
import re 
import pickle
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from spellchecker import SpellChecker
import numpy as np
import keyword
import math

PATH = "tensorflow"
file = open(PATH+"_code.pkl",'rb')
codeBlocks = pickle.load(file)
file = open(PATH+"_comment.pkl",'rb')
commentBlocks = pickle.load(file)
file = open(PATH+"_fsl.pkl",'rb')
ftSignatures = pickle.load(file)


def cleanUp(comments):
    #uses regex as well as custom defined stop words to remove unwanted words
    #takes as parameters "comments" which is a string
    #returns a list of words
    uselessRegex = r'[^\w]+'
    c = re.sub(uselessRegex,' ',comments)
    stop_words = ['the','is','are','am','was','there','at','on','where','was','how','which','be','can','for','an','in','of','off','The','but','or','their','they','can','that','they']
    words = nltk.word_tokenize(c) 
    filtered_sentence = [w for w in words if (not w in stop_words and not w.isdigit())] 
    return filtered_sentence

 
def NormalPOSTag(tokenizedComment):
    #creates taggedComment which is a list of POS tags corresponding to words in "tokenizedComment" which is a list of words
    taggedComment = nltk.pos_tag(tokenizedComment)
    return taggedComment

def PythonPOSTag(tokenizedComment):
    #similar behavior to NormalPOSTag but uses PythonPOS instead of normal POS
    #takes as parameter "tokenizedComment" which is a list
    #returns tupleList which is a python tag list
    tupleList = []
    PythonTypes = ['NoneType','int','long','float','double','complex','bool','str','string','unicode','array','ndarray','list','dataframe','tuple','set','map','dict','arr','shape','type']
    PythonKeywords = keyword.kwlist
    pythonRegexList = [r'.(A|a)rray.',r'.(N|n)one.',r'.(E|e)rror.',r'(I|i)nteger']
    for w in tokenizedComment:
        if(w in PythonTypes or w in PythonKeywords):
             tupleList.append((w,'PythonPos'))
        else:
            temp = "NotPythonPos"
            for pattern in pythonRegexList:
                if(re.match(pattern,w)):
                    temp = "PythonPos"
            tupleList.append((w,temp))
    return tupleList


def LexiconTag(tokenizedComment):
    #Creates a list containing either "special" or "lexicon" indicating if the word is out of lexicon or not
    #takes as parameter "tokenizedComment" which is a list of words (strings)
    #returns "tupleSet" which is a list of lexicon tags corresponding to the words in "tokenizedComment"
    spell = SpellChecker()
    specialWord = spell.unknown(tokenizedComment)
    for w in tokenizedComment:
        if len(w)<=2:
            specialWord.add(w)
    tupleSet = []
    for w in tokenizedComment:
        if w.casefold() in map(str.casefold, specialWord):
            tupleSet.append((w,'special'))
        else:
            tupleSet.append((w,'lexicon'))
    return tupleSet


def TFScore(word,doc):
    #Calculates the term frequency score of a "word" which is a paramater string and "doc" which is a list of words and a parameter itself(doc is usually the comment block)
    #returns a number which is the TF score
    num = 0
    for w in doc:
        if word==w[0]:
            num+=1
    return (math.log10(1+num))

def IDFScore(word,docs):
    #Calculates the inverse document frequency score where "word" and "doc" are parameters used similarly as in TFScore
    #returns a number which is the IDF score
    i=0
    for d in docs:
        for vec in d:
            if word==vec[0]:
                i+=1
                break
    return (math.log10(len(docs)/i))





