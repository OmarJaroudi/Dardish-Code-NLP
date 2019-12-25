# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:23:42 2019

@author: Omar Al Jaroudi
"""

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
    uselessRegex = r'[^\w]+'
    c = re.sub(uselessRegex,' ',comments)
    stop_words = ['the','is','are','am','was','there','at','on','where','was','how','which','be','can','for','an','in','of','off','The','and','but','or','their','they','can','that','they']
    words = nltk.word_tokenize(c) 
    filtered_sentence = [w for w in words if (not w in stop_words and not w.isdigit())] 
    return filtered_sentence

def cleanUp2(words):
    stop_words = ['the','is','are','am','was','there','at','on','where','was','how','which','be','can','for','an','in','of','off','The','and','but','or','their','they','can','that','they']
    filtered_sentence = [w for w in words if (not w in stop_words and not w.isdigit())] 
    return filtered_sentence
 
def NormalPOSTag(tokenizedComment):
    taggedComment = nltk.pos_tag(tokenizedComment)
    return taggedComment

def PythonPOSTag(tokenizedComment):
    
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


def SurroundedByKeyword(codeBlocks):
    for block in codeBlocks:
        for i,c in enumerate(block):
            if i>=2 and i<len(block)-2:
                for j in range(-2,3):
                    if block[i+j][3]=='PythonPos':
                        if len(block[i])==5:
                            block[i].append("Surrounded")
                        elif len(block[i])==6:
                            block[i][5] = "Surrounded"
                    else:
                        if len(block[i])==5:
                            block[i].append("NotSurrounded")
            elif i==0:
                for j in range(0,3):
                    if block[i+j][3]=='PythonPos':
                        if len(block[i])==5:
                            block[i].append("Surrounded")
                        elif len(block[i])==6:
                            block[i][5] = "Surrounded"
                    else:
                        if len(block[i])==5:
                            block[i].append("NotSurrounded")
            elif i==1:
                for j in range(-1,3):
                    if block[i+j][3]=='PythonPos':
                        if len(block[i])==5:
                            block[i].append("Surrounded")
                        elif len(block[i])==6:
                            block[i][5] = "Surrounded"
                    else:
                        if len(block[i])==5:
                            block[i].append("NotSurrounded")
            elif i==len(block)-2:
                for j in range(-2,2):
                    if block[i+j][3]=='PythonPos':
                        if len(block[i])==5:
                            block[i].append("Surrounded")
                        elif len(block[i])==6:
                            block[i][5] = "Surrounded"
                    else:
                        if len(block[i])==5:
                            block[i].append("NotSurrounded")
            elif i==len(block)-1:
                for j in range(-2,0):
                    if block[i+j][3]=='PythonPos':
                        if len(block[i])==5:
                            block[i].append("Surrounded")
                        elif len(block[i])==6:
                            block[i][5] = "Surrounded"
                    else:
                        if len(block[i])==5:
                            block[i].append("NotSurrounded")
    return codeBlocks

def TFScore(word,doc):
    num = 0
    for w in doc:
        if word==w[0]:
            num+=1
    return (math.log10(1+num))

def IDFScore(word,docs):
    i=0
    for d in docs:
        for vec in d:
            if word==vec[0]:
                i+=1
                break
    return (math.log10(len(docs)/i))


file = open("commentDataSet.pkl","rb")
commentSet = pickle.load(file)
file.close()

sklearnList = []
for i,comment in enumerate(commentBlocks):
    firstList = []
    comment = cleanUp(comment[0])
    n = NormalPOSTag(comment)
    lex = LexiconTag(comment)
    k = PythonPOSTag(comment)
    param = []
    for j,word in enumerate(comment):
        if word in ftSignatures[i][1]:
            param.append((word,"P"))
        else:
            param.append((word,"W"))
    
        firstList.append([word,n[j][1],param[j][1],k[j][1],lex[j][1]])

    sklearnList.append(firstList)

sklearnList = SurroundedByKeyword(sklearnList)

for c in sklearnList:
    commentSet.append(c)

AllWords = []
for c in commentSet:
    for word in c:
        AllWords.append(word[0])
        
AllWords = list(set(AllWords))
IDFScoreVector = {}
for w in AllWords:
    idf = IDFScore(w,commentSet)
    IDFScoreVector[w] = idf

for c in commentSet:
    for word in c:
        tf = TFScore(word[0],c)
        idf = IDFScoreVector[word[0]]
        total = tf*idf
        if len(word)==7:
            word[6] = total
        elif len(word)==6:
            word.append(total)



