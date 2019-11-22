"""
Created on Thu Nov 21 18:17:07 2019

@author: Mohamad Abou Harb
"""
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
