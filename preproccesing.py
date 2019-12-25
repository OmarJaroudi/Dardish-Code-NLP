# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:49:38 2019

@author: Omar Al Jaroudi
"""

import re
import pickle
import os
import numpy as np
import nltk

def SeperateCode(s):
    regex = re.compile(r'(\t*def\s*.*\(|\t*class\s*.*\()',re.M)
    blocks = [(m.start(0),m.end(0)) for m in re.finditer(regex,s)]
    if len(blocks)==0:
        return []
    indivBlocks = []
    for i in range(len(blocks)-1):
        indivBlocks.append(s[blocks[i][0]:blocks[i+1][0]])
    return indivBlocks

def disectBlock(block):
    commentIndices = [(m.start(0),m.end(0)) for m in re.finditer('\"\"\"|\'\'\'',block)]
    commentBlocks = []
    if len(commentIndices)==0 or len(commentIndices)%2!=0:
        return block,[]
    for i in range(0,len(commentIndices),2):
        commentBlocks.append(block[commentIndices[i][0]:commentIndices[i+1][1]])
    
    for c in commentBlocks:
        block = block.replace(c,'')
    
    return block,commentBlocks

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
    if s==None:
        return None
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



directory = os.fsencode("./tensorflow")

rawCode = np.array([])
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".py"):
        file = open("./tensorflow/"+filename,'r')
        rawText = file.read()
        blocks = SeperateCode(rawText)
        if len(blocks)<50:
            continue
        else:
            rawCode = np.append(rawCode,blocks)
     else:
         continue

cleanerCode = []
cleanerComment = []
ftSignatures = []
for codeBlock in rawCode:
    code,comment = disectBlock(codeBlock)
    ftSig = extractFtSignature(codeBlock)
    if ftSig!=None:
        ftSignatures.append(ftSig)
        cleanerCode.append(codeBlock)




codes = []
comments = []
cleanerSig = []
for i,codeBlock in enumerate(cleanerCode):
    codeBlock = str(codeBlock)
    code,comment = disectBlock(codeBlock)
    good = True
    for p in ftSignatures[i][1]:
        for c in comment:
            if p not in c:
                good = False
                break
    if good ==True and comment!=[] and len(comment[0])>50 and ftSignatures[i][1]!=['']:
        codes.append(code)
        comments.append(comment)
        cleanerSig.append(ftSignatures[i])


#file = open("tensorflow_code.pkl",'wb')
#pickle.dump(codes,file)
#file.close()
#
#
#file = open("tensorflow_comment.pkl",'wb')
#pickle.dump(comments,file)
#file.close()
#
#
#file = open("tensorflow_fsl.pkl",'wb')
#pickle.dump(cleanerSig,file)
#file.close()

