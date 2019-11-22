# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:49:38 2019

@author: Omar Al Jaroudi
"""

import re
import pickle
import os

def SeperateCode(s):
    regex = re.compile(r'(^\t*def\s*.*\(|^\t*class\s*.*\()',re.M)
    blocks = [(m.start(0),m.end(0)) for m in re.finditer(regex,s)]
    if len(blocks)==0:
        return []
    indivBlocks = []
    for i in range(len(blocks)-1):
        indivBlocks.append(s[blocks[i][0]:blocks[i+1][0]])
    return indivBlocks

def disectBlock(block):
    commentIndices = [(m.start(0),m.end(0)) for m in re.finditer('"""',block)]
    commentBlocks = []
    if len(commentIndices)==0 or len(commentIndices)%2!=0:
        return block,[]
    for i in range(0,len(commentIndices),2):
        commentBlocks.append(block[commentIndices[i][0]:commentIndices[i+1][1]])
    
    for c in commentBlocks:
        block = block.replace(c,'')
    
    return block,commentBlocks



directory = os.fsencode("keras")
codeList = []
commentList = []
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".py"):
        print(filename)
        file = open("keras/"+filename,'r')
        rawText = file.read()
        blocks = SeperateCode(rawText)
        if len(blocks)==0:
            continue
        for i in range(len(blocks)):
            code,comments = disectBlock(blocks[i])
            if(len(comments) > 0 and code.split(' ')[0]!='class'):
                codeList.append(code)
                commentList.append(comments)
from main import pickleObject


