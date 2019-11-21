import ast

class FunctionSignatureExtractor:
    def getFunctionArguments(self,code:str)->list:
        return [a.arg for a in ast.parse(code).body[0].args.args]
    
    def getFunctionName(self,code:str)->str:
        return ast.parse(code).body[0].name
    
    def getSignature(self,code:str)->tuple:
        return (self.getFunctionName(code),self.getFunctionArguments(code))