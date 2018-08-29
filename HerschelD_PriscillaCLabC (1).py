
# coding: utf-8

# In[60]:

'''Herschel Darko
   Priscilla Coronado
   Lab C'''



from copy import deepcopy
import operator
import random
import math
#next create choose best feature and plularity then test


            
            
        
        
    

class DecisionTree:
    #This class acts slightly different based on wheter it is given a file by user or datalist
    #by its  Accuracy function.
    def __init__(self,file=None,datalist=None):
        self.targets=set()
        self.features={}
        if (datalist==None):
            self.data=self.convert(file)
            self.validate=False
        else:
            self.validate=True
            self.data=datalist  
        self.names=self.data[1]
        self.featurevalues(self.data[0],self.data[2])
        if(file!=None):
            self.tree=self.create_tree(self.data[0],self.data[2],[])
            self.tree.display(0)
            print("Number of nodes in decision tree: ",self.tree.node_count())
            self.Accuracy(file)
        
   
    def Accuracy(self,file):
        module=[]
        with open (file, 'r') as f:
            for line in f:
                module.append([n for n in line.strip().split('\t')])
            titles=module.pop(0)
            numoffeatures=len(titles)-2
            featureIndex=(x for x in range(0,numoffeatures+1))
            featureIndex=list(featureIndex)
            datalist=(module,titles,featureIndex)
            Original=DecisionTree(None,datalist)
            #After building the orighinal dataset,we test the possible datasets combinations with a single field
            #removed and detememine if the possible dataset classify the removed field correvtly
            #the function then return the Accuracy as a value between 0 and 1
            correctpredicts=0
            n=len(datalist[0])
            for index in range(n):
                datacopy=deepcopy(datalist[0])
                currfield=datacopy.pop(index)
                currentDecisionTree=DecisionTree(None,(datacopy,titles,featureIndex))
                otarget=Original(currfield)
                ctarget=currentDecisionTree(currfield)
                if(otarget==ctarget):
                    correctpredicts+=1
            print("Test Set Accuracy: ",correctpredicts/n)       
        
            
    
    def featurevalues(self,fields,featureIndex):
        #This function updates the feature dictionary with all possible values for each feature as
        #derivived from the list of examples(fields) given.Sine a set is used the is no repeatition of 
        #values for a feature.
        targetindex=len(self.data[1])-1
        for index in featureIndex:
            self.features[index]=set()
        for field in fields:       
            self.targets.add(field[targetindex])
            for index in featureIndex:
                self.features[index].add(field[index])
                
        
        
        
     
    def addfeaturevalue(self,featIndex,value):
        self.features[featIndex].add(value)
        
    def __call__(self, field):
        #this function attempts to classify the example given to it with the  class returning the
        #target value gived by the decis
        if (self.validate):
            numoffeatures=len(field)-2
            featureIndex=(x for x in range(0,numoffeatures+1))
            for feat in featureIndex:
                self.features[feat].add(field[feat])
            self.tree=self.create_tree(self.data[0],self.data[2],[])
            return self.tree(field)
        else:
            return self.tree(field)
                       
    
    def convert(self,file):
        #this function takes the contents of a dataset file and turns it into a list of feaatures
        #a list of indexs numbers which are features and the fields of the parent node which
        #is None as these values are for the root node
        module=[]
        with open (file, 'r') as f:
            for line in f:
                module.append([n for n in line.strip().split('\t')])
        titles=module.pop(0)
        numoffeatures=len(titles)-2
        featureIndex=(x for x in range(0,numoffeatures+1))
        featureIndex=list(featureIndex)
        return (module,titles,featureIndex)

    #branch node,leaf_node

    def create_tree(self,fields,featureindexes,parent_fields):
        if not (fields):
            return self.plurality(parent_fields)
        elif self.same_class(fields):
            targetindex=len(fields[0])-1
            return fields[0][targetindex]
        elif (not(featureindexes)):
            return self.plurality(fields)
        else:
            A=self.bestfeaturetosplit(featureindexes,fields)
            Dectree=Branch(A,self.names[A])
            branches=self.splitbyfeature(A,fields)
            for branch in branches:
                newindexes=deepcopy(featureindexes)
                newindexes.remove(A)
                builtbranch=self.create_tree(branch[1],newindexes,fields)
                Dectree.grow(branch[0],builtbranch)
            return Dectree
                
                    

    def splitbyfeature(self,feature,fields):
        results=[]
        values=self.features[feature]
        for val in values:
            vlist=[]
            for field in fields:
                if (field[feature]==val):
                    vlist.append(field)
            results.append((val,vlist))
        return results

    def plurality(self,fields):
        targetcount={}
        for target in self.targets:
            targetcount[target]=0
        for field in fields:
            valindex=len(self.data[1])-1
            val=field[valindex]
            targetcount[val]+=1            
        popclass=max(targetcount.items(), key=operator.itemgetter(1))
        popclasses=[]
        for count in targetcount.items():
            if count[1]==popclass[1]:
                popclasses.append(count)
        if(len(popclasses)!=1):
            return(random.choice(popclasses))[0] 
        else:
            return popclass[0]

    
    def entropy(self,fields):
        targetfreq={}
        targ_entropy = 0.0
        targ=len(self.names)-1
        for field in fields:
            if (field[targ] in targetfreq):
                targetfreq[field[targ]] += 1.0
        else:
            targetfreq[field[targ]]= 1.0
        for freq in targetfreq.values():
            targ_entropy += (-freq/len(fields)) * math.log(freq/len(fields),2) 
        return targ_entropy
   
    
                    
    def information_gain(self,featureindex,fields):
     
        featentropy=0.0
        featval={}
       
        for field in fields:
            if (field[featureindex] in featval):
                featval[field[featureindex]]+=1
            else:
                featval[field[featureindex]]=1
            #now to get the entropy
        for feat in featval.keys():
            featprob= featval[feat]/sum(featval.values())
            featfields=[field for field in fields if (field[featureindex]==feat)]
            featentropy+=featprob*self.entropy(featfields)
        return (self.entropy(fields)-featentropy)
                
    
    def same_class(self,fields):
        target=fields[0][len(fields[0])-1]
        issame=all(field[len(fields[0])-1]==target for field in fields)
        
    def bestfeaturetosplit(self,featureindexes,fields):
        featinfos=[(self.information_gain(feat,fields),feat) for feat in featureindexes]
        bfeat=max(featinfos, key=operator.itemgetter(0))[1]
        return bfeat
        
        
        


                    
class Branch:
    
    def __init__(self, feature, featurename):
        self.feature = feature
        self.featurename = featurename
        self.children = {}
        
    def node_count(self):
         num=len(self.children)
         for value in self.children.values():
            if isinstance(value,str):
                num-=1
            else:
                #print(self.featurename,"for node at depth",depth)
                num+=value.node_count()
         return num
                
    

    def __call__(self, field):
        featurevalue = field[self.feature]
        if featurevalue in self.children:
            val=self.children[featurevalue]
            if isinstance(val,str):
                return val
            else:
                return self.children[featurevalue](field)
                    
    def grow(self,featurevalue,branch):
        self.children[featurevalue]=branch

    def display(self,count,indent=0):
        name = self.featurename
        count+=1
        for (featval,valbranch) in self.children.items():
            print(' ' * 4 * indent, name, '=',featval, end=' ')
            if (isinstance(valbranch,str)):
                print(":",valbranch)
            else:
                print("")
                count+=1
                valbranch.display(count,indent + 1)      
      
            
            
A=DecisionTree("pets.txt")



# In[ ]:




# In[ ]:



