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
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from matplotlib import pyplot as plt
##
##PATH = "./Algorithms_Pickles"
##directory = os.fsencode(PATH)
##codeBlocks = []
##commentBlocks = []
##for file in os.listdir(directory):
##     filename = os.fsdecode(file)
##     if filename.endswith(".pkl"):
##        if (filename.find("CODE")!=-1):
##            file = open(PATH+"/"+filename,'rb')
##            codeBlocks.append(pickle.load(file))
##        if (filename.find("COMMENT")!=-1):
##            file = open(PATH+"/"+filename,'rb')
##            commentBlocks.append(pickle.load(file))
##            
##assert(len(codeBlocks)==len(commentBlocks)) #sanity check
##f = open("Algorithms_CODE.pkl","wb")
##pickle.dump(codeBlocks,f)
##f.close()
##
##f = open("Algorithms_COMMENTS.pkl","wb")
##pickle.dump(commentBlocks,f)
##f.close()


PATH = "AbouHarb"
file = open(PATH+"_CODE.pkl",'rb')
codeBlocks = pickle.load(file)
file = open(PATH+"_COMMENTS.pkl",'rb')
commentBlocks = pickle.load(file)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

#print(commentBlocks[50])
#print("*************************************")
#print(codeBlocks[50])
#i=0
#j=0
#for i,c in enumerate(commentBlocks): 
#    if len(c)>100:
#        j+=1
#    else:
#        commentBlocks.remove(c)
#        codeBlocks.remove[codeBlocks[i]]

def cleanUp(comments):
    uselessRegex = r'[^\w]+'
    c = re.sub(uselessRegex,' ',comments)
    stop_words = set(stopwords.words('english')) 
    words = nltk.word_tokenize(c) 
    filtered_sentence = [w for w in words if not w in stop_words] 
    return filtered_sentence

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
        if grams[0]=='Parameters':
            paramGram = grams
            break
    return paramGram

def getGramLength(words,parameters):
    for n in range(2,len(words)-2):
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
            return n
goodWords = []
av = 0
m = -1
ratio = []
for i in range(len(commentBlocks)):
    comment = commentBlocks[i]
    code = codeBlocks[i]
    words = cleanUp(comment)
    if len(words)<20:
        continue
    res = extractFtSignature(code)
    if res==None:
        continue
    ftName = res[0]
    parameters = res[1]
    l = getGramLength((words),parameters)
    if (l!=None):
        
        goodWords.append(words)
        ratio.append(len(words)/l)

plt.plot(ratio,'o')
print(len(ratio))
assert len(ratio)==len(goodWords)
for g in goodWords:
    suggested_n = int(len(g)/np.mean(ratio))+1
    print(g)
    test_nGram = getParamGram(g,suggested_n)
    
    


