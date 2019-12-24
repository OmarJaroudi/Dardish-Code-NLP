# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import enchant
import pickle
import nltk
import os
import re
from nltk.tokenize import RegexpTokenizer
from spellchecker import SpellChecker
from PyDictionary import PyDictionary
from nltk.stem import WordNetLemmatizer 
import keyword
  
def pickleObject(object,filename):
    f = open(filename+".pkl",'wb')
    pickle.dump(object,f)
    f.close()

def lemmatizeAllComments(fileBlock):
    
    lem = WordNetLemmatizer() 
    lemmList = []
    for comment in fileBlock:
        currentWordList = []
        for w in comment:
            lemWord = lem.lemmatize(w)
            if(lemWord == ''):
                currentWordList.append(w)
            else:
                currentWordList.append(lemWord)
        lemmList.append(currentWordList)
    
    return lemmList
            
            
def IsLexicon (commentBlock):

    tokenizer = RegexpTokenizer("[^(\s|\'|\"|\`|\-|\%|\*|\+|\.|\[|\]|\/|\:|\,\=\>\<\)\(\^\}\{\~]+")
    Tokenized_Block = tokenizer.tokenize(commentBlock)


    for j,t in enumerate(Tokenized_Block):
        if(t.isnumeric()):
            Tokenized_Block[j] = 'NAN'
       
    y = ''
    
    for t in Tokenized_Block:
        if(t != 'NAN'):
            y = y + ' ' + t
            if(t == 'References'):
                break

    tokenizer = RegexpTokenizer("[a-zA-Z]+\_[a-zA-Z]*")

    Tokenized_Block1 = tokenizer.tokenize(y)
    
    tokenizer = RegexpTokenizer("[a-zA-Z]*[0-9]+[a-zA-Z]*")
    
    Tokenized_Block2 = tokenizer.tokenize(y)
    for w in Tokenized_Block2:
        Tokenized_Block1.append(w)
    
    tokenizer = RegexpTokenizer("\s[a-zA-Z][a-z]+[A-Z][a-zA-Z]*")
    
    Tokenized_Block2 = tokenizer.tokenize(y)
    for w in Tokenized_Block2:
        Tokenized_Block1.append(w)
    
    tokenizer = RegexpTokenizer("\s[a-zA-Z]\s")
    
    Tokenized_Block2 = tokenizer.tokenize(y)
    for w in Tokenized_Block2:
        Tokenized_Block1.append(w)
    
    misspelled = spell.unknown(Tokenized_Block)
    
    for word in misspelled:
        if(keyword.iskeyword(word) == False):
            Tokenized_Block1.append(word)
    
    return Tokenized_Block1
                
 
  
#print("rocks :", lemmatizer.lemmatize("rocks")) 
    


def getRawTriBiCounts(fileBlock):
    
    
    trigrams = [nltk.ngrams(fileBlock[i],3) for i in range(0,410)]
    bigrams = [nltk.ngrams(fileBlock[i],2) for i in range(0,410)]
    
    triDict =nltk.FreqDist()#dictionary containg count of all tag1 tag2 followed by P tag set
    biDict = nltk.FreqDist()
    
    for trigram in trigrams:
        triDict += nltk.FreqDist(trigram)
    for bigram in bigrams:
        biDict += nltk.FreqDist(bigram)
    return biDict,triDict


class words:
    
    word = ''
    count = 0
    
    def __init__(self, count, word):
      self.word = word
      self.count = count
      
    def ChangeCount(number):
        self.count = number
        
    def ChangeWord(newword):
        self.word = newword
      


class comparison:

    #current = words(0,'')
    #before = words(0,'')
    #after = words(0,'')
    
    def __init__ (self, before, after):  
        self.before = before
        self.after = after
    

class SetOfWords:
    
    list = []
    
    list.append(comparison(words('',0),words('',0)))



spell = SpellChecker()

# find those words that may be misspelled
#misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

misspelled = spell.unknown(['xc','xa'])



for word in misspelled:
    # Get the one `most likely` answer
    #print(spell.correction(word))
    print(word)
    # Get a list of `likely` options
    #print(spell.candidates(word))
    
#print("hello")
    
code = open("scipy_codeblocks.pkl",'rb')
codeBlocks = pickle.load(code)
#print(codeBlocks[431])
#print("*********************NEW PRINT***************************")
comments = open("scipy_comments.pkl",'rb')
commentBlocks = pickle.load(comments)

testing = open("scipy_fse.pkl",'rb')
testingBlock = pickle.load(testing)

fil = open("scipy_comments_tokenized.pkl",'rb')
fileBlock = pickle.load(fil)
#print(commentBlocks[50])


#txt = "The rain in Spain"
#x = re.search("The", txt)



i = 329

LIST = IsLexicon(commentBlocks[i])

print(LIST)


tokenizer = RegexpTokenizer("[^(\s|\'|\"|\`|\-|\%|\*|\+|\.|\[|\]|\/|\:|\,\=\>\<\)\(\^\}\{\~]+")
Tokenized_Block = tokenizer.tokenize(commentBlocks[i])


for j,t in enumerate(Tokenized_Block):
    if(t.isnumeric()):
        Tokenized_Block[j] = 'NAN'

#print(Tokenized_Block)
#print("*********************NEW PRINT***************************")
#print(Tokenized_Block2)
#print("*********************NEW PRINT***************************")
#print(codeBlocks[i])
#print("*********************NEW PRINT***************************")
#print(commentBlocks[i])

y = ''
for t in Tokenized_Block:
    if(t != 'NAN'):
        y = y + ' ' + t
    if(t == 'References'):
         break


#print("*********************NEW PRINT***************************")

#print(y)

#tokenizer = RegexpTokenizer("[^(\s|\0-9)]+")
#Tokenized_Block2 = tokenizer.tokenize(y)

tokenizer = RegexpTokenizer("[a-zA-Z]+\_[a-zA-Z]*")

Tokenized_Block1 = tokenizer.tokenize(y)

#print("*********************NEW PRINT***************************")
#print("Words with an underscore: ")
#print(Tokenized_Block1)


tokenizer = RegexpTokenizer("[a-zA-Z]*[0-9]+[a-zA-Z]*")

Tokenized_Block2 = tokenizer.tokenize(y)

#print("*********************NEW PRINT***************************")
#print("Words with numbers in them: ")
#print(Tokenized_Block2)

tokenizer = RegexpTokenizer("\s[a-zA-Z][a-z]+[A-Z][a-zA-Z]*")

Tokenized_Block3 = tokenizer.tokenize(y)
#print("Words with a capital letter inside them: ")
#print(Tokenized_Block3)


tokenizer = RegexpTokenizer("\s[a-zA-Z]\s")
Tokenized_Block5 = tokenizer.tokenize(y)
#print("Single letter words: ")
#print(Tokenized_Block5)

#add here python words
dictionary=PyDictionary()
#spell.word_frequency.load_words(dictionary)
tokenizer = RegexpTokenizer("[^(\s)]+")
Tokenized_Block4 = tokenizer.tokenize(y)
misspelled = spell.unknown(Tokenized_Block)

#print("\nThe list below are miss spelled words: \n")
#for word in misspelled:
 #   if(keyword.iskeyword(word) == False):
  #      print(word)
   
    
#print("*********************NEW PRINT***************************")
#print(testingBlock[i])
#print("*********************NEW PRINT***************************")
#print(fileBlock[i])




#lemList = lemmatizeAllComments(fileBlock)

#bi,tri = getRawTriBiCounts(lemList)

#finalList = []
#tempList = []
#for j,file in enumerate(fileBlock):
 #   tempList = []
  #  for k in range(len(file)):     
   #     l = []
    #    first = nltk.pos_tag([file[k]])
     #   second = [lemList[j][k]]
      #  l.append([file[k],lemList[j][k],first[0][1]])
       # tempList.append(l)
        
    #finalList.append(tempList)
    


#pickleObject(finalList,"FinalList")



