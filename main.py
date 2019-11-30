"""
Created on Thu Nov 21 18:17:07 2019

@author: Mohamad Abou Harb
"""
import pandas as pd
import re 
import pickle
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from utils import FunctionSignatureExtractor

def loadPickleFile(path:str) -> list:
    try:
        file = open(path,'rb')
    except:
        print('File not found.')
        return
    return pickle.load(file)

def cleanUpComments(comments:list) -> list:
    commentsVectorList = []
    pattern = r'[^\w]+' #pattern to detect non alpha numeric characters
    
    stop_words = set(stopwords.words('english'))#load nltk stop words
    
    for comment in comments:#loop over each comment tokenizing and removing stop words
        comment = re.sub(pattern,' ',comment) # replace pattern with empty string
        currentWordVector = nltk.word_tokenize(comment)
        
        commentsVectorList.append(currentWordVector)#append filtered tokenized comment to final list
    return commentsVectorList

def removeClassBlocks(comments:list,codeBlocks:list):
    commentToRemove = []
    codeToRemove = []
    for i,code in enumerate(codeBlocks):
        if code.split(" ")[0] == "class":
            commentToRemove.append(comments[i])
            codeToRemove.append(codeBlocks[i])
    for comment in commentToRemove:
        comments.remove(comment)
    for code in codeToRemove:
        codeBlocks.remove(code)

def removeBadCodeCommentPair(comments:list,codeBlocks:list):
    fse = FunctionSignatureExtractor()
    commentToRemove = []
    codeToRemove = []
    for i,code in enumerate(codeBlocks):
        try:
            fse.getSignature(code)
        except:
            commentToRemove.append(comments[i])
            codeToRemove.append(codeBlocks[i])
    for comment in commentToRemove:
        comments.remove(comment)
    for code in codeToRemove:
        codeBlocks.remove(code)

def getFunctionSignatures(codeBlocks:list)->list:
    fse = FunctionSignatureExtractor()
    functionSignatureList = []
    for code in codeBlocks:
        functionSignatureList.append(fse.getSignature(code))
    return functionSignatureList

def pickleObject(object,filename):
    f = open(filename+".pkl",'wb')
    pickle.dump(object,f)
    f.close()
    
def getFSECommentTuple(comments:list,codeBlocks:list)->tuple:
    comments = cleanUpComments(comments)
    removeBadCodeCommentPair(comments,codeBlocks)
    codeBlocks = getFunctionSignatures(codeBlocks)
    return comments,codeBlocks

def tagComments(comments:list)->list:
    taggedComments = []
    for comment in comments:
        taggedComments.append(nltk.pos_tag(comment))
    return taggedComments

def annotateParameters(comments:list,functionSignature:list)->list:
    taggedComments = tagComments(comments)
    annotatedComments = []
    for i,comment in enumerate(taggedComments):
        currentList = []
        for t in comment:
            if t[0] in functionSignature[i][1]:
                currentList.append([t[0],'P'])
            else:
                currentList.append([t[0],t[1]])
        annotatedComments.append(currentList)
    return annotatedComments

def createPosTagList(annotatedComments:list)->list:
    posTagList = []
    for ac in annotatedComments:
        currentPosList = []
        for t in ac:
            currentPosList.append(t[1])
        posTagList.append(currentPosList)
    return posTagList

def getRawTriBiCounts(posTagList:list):
    trigrams = [nltk.ngrams(posTagList[i],3) for i in range(0,410)]
    bigrams = [nltk.ngrams(posTagList[i],2) for i in range(0,410)]
    ttp =nltk. FreqDist()#dictionary containg count of all tag1 tag2 followed by P tag set
    tt = nltk.FreqDist()
    for trigram in trigrams:
        ttp += nltk.FreqDist(trigram)
    for bigram in bigrams:
            tt += nltk.FreqDist(bigram)
    return tt,ttp

def predict(tokenizedComment,tt,ttp):
    taggedComment = nltk.pos_tag(tokenizedComment)
    pVector = []
    for i in range(2,len(taggedComment)):
        try:
            ttpKey =(taggedComment[i-2][1],taggedComment[i-1][1],taggedComment[i][1])
            ttKey = (taggedComment[i-2][1],taggedComment[i-1][1])
            if(ttKey in tt.keys() and ttpKey in ttp.keys()):
                p = ttp[ttpKey]/tt[ttKey]
                pVector.append((tokenizedComment[i],p))
            else:
                pVector.append((tokenizedComment[i],0))
        except Exception as e:
            print("here",e)
    return pVector
        
    
scipyComments = loadPickleFile('Pickle/scipy/scipy_comments_tokenized.pkl')
scipyFSE = loadPickleFile('Pickle/scipy/scipy_fse.pkl')

annotatedComments = annotateParameters(scipyComments,scipyFSE)
posTagList = createPosTagList(annotatedComments)

from operator import itemgetter

tt,ttp = getRawTriBiCounts(posTagList)
pVec = predict(scipyComments[411],tt,ttp)
pVec = sorted(pVec,key=itemgetter(1),reverse = True)
