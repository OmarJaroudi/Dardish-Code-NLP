import os
from nltk.tokenize import RegexpTokenizer
import re
import pickle


#CodeBlock holds the code
#CommentBlock holds the corresponding comments
#functionsBlock holds the strings of functions (to be made into tuple)
#Comments has everything (code and comments from original .txt line seperated)
#LineSeperated is a list of the .txt files and each item is a list of lines of the .txt file

def pickleObject(object,filename):
    f = open(filename+".pkl",'wb')
    pickle.dump(object,f)
    f.close()

directory = os.fsencode("./javat")
rawCode = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        file = open("./javat/"+filename,'r')
        rawText = file.read()
        rawCode.append(rawText)
        file.close()
        
print("here")   


tokenizer = RegexpTokenizer("[^\r\n]+")
LineSeperated = []


for i in range(len(rawCode)):
    LineSeperated.append(tokenizer.tokenize(rawCode[i]))
    

tokenizer = RegexpTokenizer("((^(\s+)(\/|\*+)+)|(^(\/+|\*+)+))")

CommentBlock = []
CodeBlock = []

Comments = []
switchBefore = 0
switchAfter = 0
for files in LineSeperated:
    stringC = ""
    stringT = ""
    test = []
    for k in files:
        Comments.append(tokenizer.tokenize(k))
        test = Comments[len(Comments)-1]
        
        
        if(len(test) == 1):
            stringC = stringC + k + " "
            switchAfter = 0
        else:
            stringT = stringT + k + " "
            switchAfter = 1
        
        
        if(switchBefore != switchAfter):
            if(switchBefore == 0):
                CommentBlock.append(stringC)
                stringC = ""
                switchBefore = switchAfter
            else:
                CodeBlock.append(stringT)
                stringT = ""
                switchBefore = switchAfter
                
if switchBefore == 0:
    CommentBlock.append(stringC)
else:
    CodeBlock.append(stringT)
        
#the below tokenizer finds everything after (public, protected, private, and static) up until a {
#use it to take out the function name and parameters
match = re.compile(r'(public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])')
p1 = re.compile("(\w+|\s+)\)") #to find last parameter before )
p2 = re.compile("(\w+|\s+)\(") #to find function name
p3 = re.compile("(\w+|\s+)\,") #to find parameters between ()
functionsBlock = [] #Main list holding all functions
CCBlock = [] #storing all corresponding comments for the functions
TempCommentBlock = CommentBlock
BetterCommentBlock = []

for j , func in enumerate(CodeBlock):
    m = match.search(func)
    items = []
    if(m != None):
        m1 = p1.search(m.group())
        m2 = p2.search(m.group())
        m3 = p3.findall(m.group())
        s = ""
        
        if m2 != None:
            s = m2.group()
            s = s.replace("(",'')
            items.append(s)
        
        if m1 != None:
            s = m1.group()
            s = s.replace(")",'')
            items.append(s)
        if m3 != None:
            for i in range(len(m3)):
                s = m3[i]
                s = s.replace(",",'')
                items.append(s)
        
        
           
    if len(items) != 0:
        functionsBlock.append(items)
        CCBlock.append(CommentBlock[j])
        BetterCommentBlock.append(TempCommentBlock[j])
        
    else:
        
        if(j != len(TempCommentBlock)-1):
            TempCommentBlock[j+1] = TempCommentBlock[j]  + " " + TempCommentBlock[j+1]
            #TempCommentBlock.remove(TempCommentBlock[j])
        
    
    
pickleObject(CommentBlock,"AllCommentsBlock")
    

            


