"""
Created on Thu Nov 21 18:17:07 2019

@author: Mohamad Abou Harb
"""

import re 
import pickle
import nltk
import keyword
import pandas as pd
import math
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from utils import FunctionSignatureExtractor
from spellchecker import SpellChecker
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image  
import pydotplus
import matplotlib.pyplot as plt

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

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
                currentList.append([t[0],t[1],'P'])
            else:
                currentList.append([t[0],t[1],'W'])
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

def reduceComments(annotatedComments:list)->list:
    reducedComments = []
    digitPattern = r'\d+'
    uselessWords = ['the','is','are','am','was','there','at','on','where','was','how','which','be','can','for','an','in','of','off','The','and','but','or','their','they','can','that','as','such']
    for ac in annotatedComments:
        current = []
        for w in ac:
            if(not re.match(digitPattern,w[0]) and w[0] not in uselessWords):
                current.append(w)
        reducedComments.append(current)
    
    return reducedComments

def getTermFrequencyVectors(reducedComments:list)->list:
    localTFVector = []
    globalTFDict = {}
    for comment in reducedComments:
        localDict = {}
        for w in comment:
            if((w[0],w[2]) in globalTFDict.keys()):
                globalTFDict[(w[0],w[2])] +=1
            else:
                globalTFDict[(w[0],w[2])] = 1
            if((w[0],w[2]) in localDict.keys()):
                localDict[w[0],w[2]] +=1
            else:
                localDict[w[0],w[2]] = 1
        localTFVector.append(localDict)
    return localTFVector,globalTFDict

def tagPythonPos(annotatedComments:list):
    pythonTypes = ['NoneType','int','long','float','double','complex','bool','str','string','unicode','array','ndarray','list','dataframe','tuple','set','map','dict','arr','shape','type']
    pythonKeywords = keyword.kwlist
    pythonRegexList = [r'.*(A|a)rray.*',r'.*(N|n)one.*',r'.*(E|e)rror.*',r'(I|i)nteger']
    for comment in annotatedComments:
        for w in comment:
            if(w[0] in pythonTypes or w[0] in pythonKeywords):
                w.append('PythonPos')
            else:
                w.append('NotPythonPos')
            for pattern in pythonRegexList:
                if(re.match(pattern,w[0])):
                    if(len(w) == 4):
                        if(w[3] == 'NotPythonPos'):
                            w[3] = 'PythonPos'
    return annotatedComments
def tagLexicon(annotatedComments):
    spell = SpellChecker()
    for ac in annotatedComments:
        tc = [w[0] for w in ac]
        specialWord = spell.unknown(tc)
    
        for w in ac:
            if len(w[0])<=2:
                specialWord.add(w[0])
        for w in ac:
            if w[0].casefold() in map(str.casefold, specialWord):
                w.append('special')
            else:
                w.append('lexicon')
    return annotatedComments
def convertToPandasDF(reducedComments):
    data = {'Word':[],'POS':[],'Python POS':[],'In-Lexicon':[],'RelativeIdx':[],'TF-IDF':[],'Output':[]}
    for comment in reducedComments:
        for w in comment:
            data['Word'].append(w[0])
            data['POS'].append(w[1])
            if(w[2] == 'P'):
                data['Output'].append(1)
            else:
                data['Output'].append(0)
            if(w[3] == 'PythonPos'):    
                data['Python POS'].append(1)
            else:
                data['Python POS'].append(0)
            if(w[4] == 'lexicon'):
                data['In-Lexicon'].append(1)
            else:
                data['In-Lexicon'].append(0)

            data['TF-IDF'].append(w[6])
            data['RelativeIdx'].append(w[7])
    return pd.DataFrame(data)
def oneHotEncode(df,col,prefix):
    df[col] = pd.Categorical(df[col])
    dfDummies = pd.get_dummies(df[col], prefix = prefix)
    df = df.drop(col,axis=1)
    df = pd.concat([df,dfDummies],axis=1)
    return df
def generateReducedCorpus(reducedComments:list)->list:
    corpus = []
    for comment in reducedComments:
        currentComment = ""
        for w in comment:
            currentComment += w[0]
            currentComment += " "
        corpus.append(currentComment)
    return corpus
scipyComments = loadPickleFile('Pickle/scipy/scipy_comments_tokenized.pkl')
scipyFSE = loadPickleFile('Pickle/scipy/scipy_fse.pkl')
def markWordPosition(annotatedComments):
    for i,comment in enumerate(annotatedComments):
        for w in comment:
            w.append(i)
    return annotatedComments

def dropCol(df,cols):
    for col in cols:
        df = df.drop(col,axis = 1)
    return df
def getCommentFetureSet(comment,df):
    relativeIndexFeature = df.RelativeIdx.max()+1
    return
annotatedComments = loadPickleFile('commentDataSet.pkl')
annotatedComments = markWordPosition(annotatedComments)





df = convertToPandasDF(annotatedComments)
#df=df.drop('TF-IDF',axis=1)
#df = MultiColumnLabelEncoder(columns = ['POS','Python POS','In-Lexicon','Word']).fit_transform(df)

from sklearn import preprocessing
posEncoder = preprocessing.LabelEncoder()
posEncoder.fit(df['POS'])
df['POS'] =posEncoder.transform(df['POS'])

wordEncoder = preprocessing.LabelEncoder()
wordEncoder.fit(df['Word'])
df['Word'] =wordEncoder.transform(df['Word'])

#print(list(posEncoder.classes_))
#print(wordEncoder.transform(["x_avg"]))
#print(list(posEncoder.inverse_transform([1])))


y = df.Output
df = df.drop('Output',axis = 1)



df['Output'] = y

feature_cols = [col for col in df.columns]
feature_cols.remove('Output')
#feature_cols.remove('Word')

X = df[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


clf = DecisionTreeClassifier(max_depth = 30)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)



print('Descion Tree')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred, average="binary"))
print("Prescion:",metrics.precision_score(y_test, y_pred, average="binary"))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph


plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] )
ax.set_yticklabels([''])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


