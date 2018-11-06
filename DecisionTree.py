
import pandas as pd
import math
import random
import copy
import sys
  
if(len(sys.argv) != 7):
    sys.exit("Please give the required amount of arguments -  <L> <K> <training-set> <validation-set> <test-set> <to-print>")
else:
#    l = 10
#    k = 10
#    pathToTrainingFile = "D:\\MS\\CS6375 ML by Anjum Chida\Assignments\\1\\data_sets1\\training_set.csv"
#    pathToTestFile = "D:\\MS\\CS6375 ML by Anjum Chida\Assignments\\1\\data_sets1\\test_set.csv"
#    pathToValidationSet = "D:\\MS\\CS6375 ML by Anjum Chida\Assignments\\1\\data_sets1\\validation_set.csv"
#    toPrint = 'no'
    l = int(sys.argv[1])
    k = int(sys.argv[2])
    trainPath = sys.argv[3]
    validationPath = sys.argv[4]
    testPath = sys.argv[5]
    toPrint = sys.argv[6]

# reading files
train = pd.read_csv(trainPath)          
test = pd.read_csv(testPath)               
validate = pd.read_csv(validationPath)      

def WhoisTheBest(dataframe, EntOrVar):
    maxIG= -100; 
    bestcolName = None;
    for x in dataframe.columns:
        if x!= 'Class':
            currentIG = calculateIG(dataframe[[x,'Class']],x, EntOrVar)
            if(maxIG < currentIG):
                maxIG = currentIG
                bestcolName = x
    return bestcolName   
 
def CalculateE(dataframe): 
    totalCount= dataframe.shape[0];
    
    ones = dataframe[dataframe['Class']==1].shape[0]
    zeroes = totalCount- ones;
    if(ones == totalCount or zeroes == totalCount):
        return 0.00
    else:
        Entropy = -(ones/totalCount)*math.log(ones/totalCount, 2) - (zeroes/totalCount)*math.log(zeroes/totalCount,2)
        return Entropy
    
def CalculateV(dataframe):
    totalCount= dataframe.shape[0];
    ones = dataframe[dataframe['Class']==1].shape[0]
    zeroes = totalCount- ones
    if(ones == totalCount or zeroes == totalCount):
        return 0.00
    else:
        variance = (ones/totalCount)*(zeroes/totalCount)
        return variance

def calculateIG(dataframe,x,EntOrVar):
    totalCount = dataframe.shape[0]
    ones = dataframe[dataframe[x]==1].shape[0]
    zeroes = totalCount - ones
    
    if(EntOrVar == 0):# 0 for Entropy and 1 for variance
        EntropyS = CalculateE(dataframe[['Class']])
        EntropyOnes = CalculateE(dataframe[dataframe[x]==1][['Class']])
        Entropyzeroes = CalculateE(dataframe[dataframe[x]==0][['Class']])
        return (EntropyS - ((EntropyOnes*(ones/totalCount) +  (Entropyzeroes*(zeroes/totalCount)))))
    else:
        EntropyS = CalculateV(dataframe[['Class']])
        EntropyOnes = CalculateV(dataframe[dataframe[x]==1][['Class']])
        Entropyzeroes = CalculateV(dataframe[dataframe[x]==0][['Class']])
        return (EntropyS - ((EntropyOnes*(ones/totalCount) +  (Entropyzeroes*(zeroes/totalCount)))))                                      



class Node():
    def __init__(self):
        self.colName = None
        self.left = None
        self.right = None
        
        self.nodeType = None         
        self.holder = None           
        self.endTag = None
        
        self.positiveCase = None
        self.negativeCase = None
        
        
        self.nID = None
    
    def createNode(self, colName, nodeType, holder = None, positiveCase = None, negativeCase = None):
        self.colName = colName
        self.nodeType = nodeType
        self.holder = holder
        
        self.positiveCase = positiveCase
        self.negativeCase = negativeCase

class decisionTree():
    
    
    def __init__(self):
        self.root = Node() 
        self.root.createNode('root','r',None, None, None)
        
        
    def createDTree(self, dataframe, node,EntOrVar):
        
        global nodeCount
        totalCount = dataframe.shape[0];
        ones = dataframe[dataframe['Class']==1].shape[0]
        zeroes= totalCount - ones
        #print (ones)
        
        if dataframe.shape[1]==1 or totalCount == ones or totalCount == zeroes:
            node.nodeType = 'l'
            if ones > zeroes:
                node.endTag= 1
            else:
                node.endTag= 0;
            return
        else:
            
            bestcolName= WhoisTheBest(dataframe, EntOrVar)
            #node.colName= bestcolName
            
            node.left = Node()
            positiveCase = dataframe[(dataframe[bestcolName]==0) & (dataframe['Class']==1)].shape[0]
            negativeCase = dataframe[(dataframe[bestcolName]==0) & (dataframe['Class']==0)].shape[0]

            
            node.left.createNode(bestcolName,'i',0,positiveCase,negativeCase)
            #print(bestcolName,positiveCase,negativeCase)
            
            
            node.right = Node()
            positiveCase = dataframe[(dataframe[bestcolName] == 1) & (dataframe['Class']==1)].shape[0]
            negativeCase = dataframe[(dataframe[bestcolName]==1) & (dataframe['Class']==0)].shape[0]
#            
            
            node.left.nID = nodeCount
            nodeCount=nodeCount + 1
            node.right.nID = nodeCount
            nodeCount=nodeCount + 1
            
            node.right.createNode(bestcolName,'i',1,positiveCase,negativeCase)
            #print(bestcolName,positiveCase,negativeCase)
            
            self.createDTree(dataframe[dataframe[bestcolName]==0].drop([bestcolName],axis=1), node.left, EntOrVar)
            self.createDTree(dataframe[dataframe[bestcolName]==1].drop([bestcolName],axis=1), node.right, EntOrVar)
                          
    def printDtree(self, root, depth):
        
        if(root.colName == 'root'):
            self.printDtree(root.left,0)
            self.printDtree(root.right,0)
        else:
            if(root.right is None and root.left is None): 
                for i in range(0,depth):    
                    print("| ",end="")
                depth = depth + 1
                if(root.endTag is not None):    
                   print(root.colName + " = " + str(root.holder) + " : " + str(root.endTag))
                else:
                   print(root.colName + " = " + str(root.holder) + " : ")  
            elif(root.left is not None and root.right is None):
                for i in range(0,depth):    
                    print("| ",end="")
                depth = depth + 1
                if(root.endTag is not None):    
                   print(root.colName + " = " + str(root.holder) + " : " + str(root.endTag))
                else:
                   print(root.colName + " = " + str(root.holder) + " : "   )
                   self.printDtree(root.left,depth)
            elif(root.left is None and root.right is not None):
                for i in range(0,depth):    
                    print("| ",end="")
                depth = depth + 1
                if(root.endTag is not None):    
                    print(root.colName + " = " + str(root.holder) + " : " + str(root.endTag))
                else:
                    print(root.colName + " = " + str(root.holder) + " : ")
                self.printDtree(root.right,depth)
            else:
                for i in range(0,depth):    
                    print("| ",end="")
                depth = depth + 1; 
                if(root.endTag is not None):
                   print(root.colName + " = " + str(root.holder) + " : " + str(root.endTag))
                else:
                   print(root.colName + " = " + str(root.holder) + " : ")
                self.printDtree(root.left,depth)
                self.printDtree(root.right,depth)
    
    def traverseTree(self,row, rowid, root):
        if root.endTag is not None:
            return root.endTag
        elif row[root.left.colName][rowid] == 1:  
            return self.traverseTree(row, rowid ,root.right)
        else:
            return self.traverseTree(row, rowid ,root.left)
            
    def countRoot(self,node):
        if(node.left is not None and node.right is not None):
            return self.countRoot(node.left) + self.countRoot(node.right) + 2
        elif(node.left is None and node.right is not None):
           return self.countRoot(node.right) + 1
        elif(node.left is not None and node.right is  None):
            return self.countRoot(node.left) + 1
        else:
            return 0

def pruning(m, Dtree):
    for j in range(1,m+1):
        n = Dtree.countRoot(Dtree.root);
        p=random.randint(1,n)
        
        node = findNodeP(p, Dtree.root)
        if(node is None):
            pass
        else:
            node.left = None
            node.right = None
            node.nodeType = "l"     
            if(node.negativeCase >= node.positiveCase):
                node.endTag = 0
            else:
                node.endTag = 1
        return Dtree
    
def findNodeP(p,root):
    flag = None
    node = None
    
    if(root.nodeType != "l"):
        if(root.nID == p):
            return root
        else:
            node = findNodeP(p,root.left)
            if (node is None):
                node = findNodeP(p,root.right)
            return node
    else:
        return flag

def accuracy(dataframe, Dtree):
    successCount = 0
    for rowID in dataframe.index:
        checkRow= dataframe.iloc[rowID:rowID+1, :]
        checkRow=checkRow.drop(['Class'],axis =1)
        predictedholder = Dtree.traverseTree(checkRow,rowID,Dtree.root)
        if predictedholder == dataframe['Class'][rowID]:
            successCount=successCount + 1
    return (round(successCount/dataframe.shape[0]*100,2))

print("Entropy Heuristic")

#create Tree
nodeCount = 0
dTree = decisionTree()
dTree.createDTree(train, dTree.root, 0)

#Accuracy on validate data set
accuracy1 = accuracy(validate, dTree)
dBest = copy.deepcopy(dTree)


if toPrint == 'yes':
    print("Printing Tree 1 before pruning")
    dTree.printDtree(dTree.root,0)
    print("Tree printed successfully")
    
print("Accuracy on Test data before pruning : " + str(accuracy(test, dTree) ))
print("Printing L,K and accuracy on validation set(Entropy) ")
for i in range(1, l + 1):                    
    dDashPrune = decisionTree()
    dDashPrune = copy.deepcopy(dBest)
    m = random.randint(1,k)
    dDashPrune = pruning(m,dDashPrune)
    accuracy2 = accuracy(validate, dDashPrune)
   
    print("(L,K) - ("+str(i) + "," + str(m) +") :" + str(accuracy2) )
    if accuracy2 > accuracy1:             
        accuracy1 = accuracy2
        dBest = copy.deepcopy(dDashPrune)
        



if toPrint == 'yes':
    print("Printing Tree 1 after pruning ")
    dBest.printDtree(dBest.root,0)
    print("Tree printed successfully")

print("Accuracy on Test data after pruning : " + str(accuracy(test, dBest) ))
print("\n")

    
print("Variance Heuristic")

nodeCount = 0
dTree2 = decisionTree()
dTree2.createDTree(train, dTree2.root, 1)

#Accuracy on validate data set
accuracy1 = accuracy(validate, dTree2)
dBest2 = copy.deepcopy(dTree2)

if toPrint == 'yes':
    print("Printing Tree 1 before pruning ")
    dTree2.printDtree(dTree2.root,0)
    print("Tree printed successfully")

print("Accuracy on Test data before pruning : " + str(accuracy(test, dTree2) ))

print("Printing L,K and accuracy on validation set (Variance)")

for i in range(1, l + 1):                    
    dDashPrune2 = decisionTree()
    dDashPrune2 = copy.deepcopy(dBest2)
    m = random.randint(1,k)
    dDashPrune2 = pruning(m,dDashPrune2)
    accuracy2 = accuracy(validate, dDashPrune2)
    print("(L,K) - ("+str(i) + "," + str(m) +") :" + str(accuracy2) )
    if accuracy2 > accuracy1:             
        accuracy1 = accuracy2
        dBest2 = copy.deepcopy(dDashPrune2)


if toPrint == 'yes':
    print("Printing Tree 1 after pruning ")
    dBest2.printDtree(dBest2.root,0)
    print("Tree printed successfully")

print("Accuracy on Test data after pruning : " + str(accuracy(test, dBest2) ))

print("\n")