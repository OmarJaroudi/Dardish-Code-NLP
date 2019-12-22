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

PATH = "scipy"
file = open(PATH+"_codeblocks.pkl",'rb')
codeBlocks = pickle.load(file)
file = open(PATH+"_comments.pkl",'rb')
commentBlocks = pickle.load(file)
file = open(PATH+"_comments_tokenized.pkl",'rb')
commentTokenized = pickle.load(file)
file = open(PATH+"_fsl.pkl",'rb')
ftSignatures = pickle.load(file)


def cleanUp(comments):
    uselessRegex = r'[^\w]+'
    c = re.sub(uselessRegex,' ',comments)
#    stop_words = set(stopwords.words('english')) 
    words = nltk.word_tokenize(c) 
#    filtered_sentence = [w for w in words if not w in stop_words] 
    return words
    
def NormalPOSTag(tokenizedComment):
    taggedComment = nltk.pos_tag(tokenizedComment)
    return taggedComment

def PythonPOSTag(tokenizedComment):
    
    tupleList = []
    n = 0
    PythonTypes = ['NoneType','int','long','float','double','complex','bool','str','string','unicode','array','ndarray','list','dataframe','tuple','set','map','dict','arr','shape','type']
    PythonKeywords = ['False','class','finally','is','return','None','continue','for','lambda','try','True','def','from','nonlocal','while','and','del','global','not','with','as','elif','if','or','yield','assert','else','import','pass','break','except','in','raise']
    for w in tokenizedComment:
        isSpecial = False
        
            
        for p in PythonTypes:
            s = re.search(p,w)
            if s!=None:
                tupleList.append((w,'type'))
                n+=1
                isSpecial = True
                break
        for p in PythonKeywords:
            s = re.search(p,w)
            if s!=None and (w,'type') not in tupleList:
                tupleList.append((w,'keyword'))
                n+=1
                isSpecial = True
                break
        if not isSpecial:
            tupleList.append((w,'ordinary'))
    return tupleList


def LexiconTag(tokenizedComment):
    spell = SpellChecker()
    specialWord = spell.unknown(tokenizedComment)
    for w in tokenizedComment:
        if len(w)==1 and w not in set(stopwords.words('english')):
            specialWord.add(w)
    tupleSet = []
    for w in tokenizedComment:
        if w.casefold() in map(str.casefold, specialWord):
            tupleSet.append((w,'special'))
        else:
            tupleSet.append((w,'lexicon'))
    return tupleSet



    
def GenerateVector(tokenizedComment):
    vec = []
    l = LexiconTag(tokenizedComment)
    n = NormalPOSTag(tokenizedComment)
    p = PythonPOSTag(tokenizedComment)
    
    for i,w in enumerate(tokenizedComment):
        vec.append((w,n[i][1],p[i][1],l[i][1]))
    return vec

"""         
specialParam = 0
Omega = 0

for i in range(len(commentBlocks)):
    c = cleanUp(commentBlocks[i])
    vec = GenerateVector(c)
    for v in vec:
        if v[0] in ftSignatures[i][1]:
            Omega+=1
            if v[3]=='special':
                specialParam+=1
    
print(specialParam/Omega)
"""
