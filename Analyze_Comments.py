# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:36:30 2019

@author: Omar Al Jaroudi
"""
import pickle
import nltk
import re
import numpy as np
from nltk.corpus import stopwords 
from nltk import ngrams
import scipy
from Distribution_Fit import Distribution
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
dist = Distribution()
from matplotlib import pyplot as plt
import pandas as pd
from operator import itemgetter

PATH = "scipy"
file = open(PATH+"_codeblocks.pkl",'rb')
codeBlocks = pickle.load(file)
file = open(PATH+"_comments.pkl",'rb')
commentBlocks = pickle.load(file)
file = open(PATH+"_comments_tokenized.pkl",'rb')
commentTokenized = pickle.load(file)
file = open(PATH+"_fsl.pkl",'rb')
ftSignatures = pickle.load(file)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]



def cleanUp(comments):
    uselessRegex = r'[^\w]+'
    c = re.sub(uselessRegex,' ',comments)
#    stop_words = set(stopwords.words('english')) 
    words = nltk.word_tokenize(c) 
#    filtered_sentence = [w for w in words if not w in stop_words] 
    return words

def extractFtSignature(code):
    signatureRegex = r'def.*\:'
    c = code
    s = re.search(signatureRegex,c)
    if s==None:
        return None
    ftSignature = c[s.start(0):s.end(0)]
    ftNameRegex = r'\s+[a-zA-Z_].*\('
    ftName = re.search(ftNameRegex,ftSignature)
    if ftName==None:
        return None
    ftName = ftSignature[ftName.start(0):ftName.end(0)-1]
    ftName = ftName.strip()

    parametersRegex = r'\(.*\)'
    s = re.search(parametersRegex,ftSignature)
    parameters = ftSignature[s.start(0)+1:s.end(0)-1].split(',')
    for i,p in enumerate(parameters):
        temp = ""
        for letter in p:
            if letter !='=':
                temp+=letter
            else:
                break
            parameters[i] = temp.strip()
    return (ftName,parameters)

def getParamGram(words,n=2):
    nGrams = ngrams(words, n)
    paramGram = None
    for grams in nGrams:
        regex = '[pP]aram[eters]?'
        m = re.search(regex,grams[0])
        if m!=None:
            paramGram = grams
            break
    return paramGram

def getGramLength(words,parameters):
    for n in range(2,len(words)-1):
        gramOfInterest = getParamGram(words,n)
        Found = True
        if gramOfInterest==None:
            continue
        for p in parameters:
            if p not in gramOfInterest:
                Found = False
                break
            else:
                Found = True
        if Found == True:
            return n+1

paramSynonym = ['param(eter(s)?)?','arg(ument(s)?)?','var(iable(s)?)?','input','takes','accepts','expects']
returnSynonym = ['return(s)?','output(s)?','prints','riase(s)?','throw(s)?','exception']


def getIndexPositions(listOfElements, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    indexPosList = []
    indexPos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            indexPos = listOfElements.index(element, indexPos)
            # Add the index position in list
            indexPosList.append(indexPos)
            indexPos += 1
        except ValueError as e:
            break
    return indexPosList

def annotateComments(blocks,ftSignatures):
    annotatedWords = []
    for i,c in enumerate(blocks):
        if i==400:
            break
        c = cleanUp(c)
        for word in c:
            if word in ftSignatures[i][1]:
                annotatedWords.append((word,'P'))
            else:
                annotatedWords.append((word,'W'))
    return (annotatedWords)
#
#file = open("annotated.pkl",'rb')
#annotatedComments = pickle.load(file)
#file.close()
#
#tokenizedComments = []
#for i,c in enumerate(commentBlocks):
#    if i==400:
#        break
#    c = cleanUp(c)
#    for word in c:
#        tokenizedComments.append(word)
#
#param = []
#for ann in annotatedComments:
#    if ann[1]=='P':
#        param.append(ann[0])
#
#param = list(set(param))
#
#biGrams = nltk.ngrams(tokenizedComments,2)
#biGrams = list(set(biGrams))
#stringRep = []
#for b in biGrams:
#    stringRep.append(str(b[0]+','+str(b[1])))
#df = pd.DataFrame(1,index = stringRep,columns = param)
#
#def findWordsBeforeParameter(param:str,commentSet):
#    idx = getIndexPositions(commentSet,param)
#    wordPairs =[] 
#    for i in idx:
#        wordPairs.append(str(commentSet[i-2])+','+str(commentSet[i-1]))
#    return wordPairs
#
#for p in param:
#    
#    wordPairs = findWordsBeforeParameter(p,tokenizedComments)
#    for w in wordPairs:
#        df.at[w,p]+=1
#
#newbiGrams = nltk.ngrams(tokenizedComments,2)
#fdist = nltk.FreqDist(newbiGrams)
#V = len(list(set(tokenizedComments)))
#for key in fdist.keys():
#    stringKey = str(key[0])+','+str(key[1])
#    df.loc[stringKey,:]/=(fdist[key]+V)


file = open('Dataframe.pkl','rb')
df = pickle.load(file)
file.close()
V = len(list(set(tokenizedComments)))

df['sum'] = df.sum(axis=1)
SUM = df.loc[:,'sum'].values
def predictParameters(commentBlock):
    commentBlock = cleanUp(commentBlock)
    candidates = []
    for i in range(2,len(commentBlock)):
        w1 = commentBlock[i-2]
        w2 = commentBlock[i-1]
        stringKey = w1+','+w2
        probability=0
        try:
            probability = df.loc[stringKey,'sum']
        except:
            probability = 1/V
        candidates.append((commentBlock[i],probability))
    candidates = list(set(candidates))
    return sorted(candidates,key=itemgetter(1),reverse = True)

i = 504
print(predictParameters(commentBlocks[i]))
print(commentBlocks[i])
print(ftSignatures[i])
           

