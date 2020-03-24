#code for Question 2
#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
#importing the tic tac toe data using the link
url='https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'
d=pd.read_csv(url)
d=np.array(d)
#Separating the features and labels
y=d[:,9]
x=d[:,:9]
#importing the wine data using the link
url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
w=pd.read_csv(url)
w=np.array(w)
#Separating the labels and features
w_y=w[:,0]
w_x=w[:,1:]
#Function to compute the confusion matrix for the wine dataset
def wine_confusion_matrix(t_x,t_y,tree):
    pred_list=[]
    real_list=[]
    for i in range(t_x.shape[0]):
        if wine_pred(tree,t_x[i])==1:
            pred_list.append(1)
            if (t_y[i])==1:
                real_list.append(1)
            if (t_y[i])==2:
                real_list.append(2)
            if (t_y[i])==3:
                real_list.append(3)
        if wine_pred(tree,t_x[i])==2:
            pred_list.append(2)
            if (t_y[i])==1:
                real_list.append(1)
            if (t_y[i])==2:
                real_list.append(2)
            if (t_y[i])==3:
                real_list.append(3) 
        if wine_pred(tree,t_x[i])==3:
            pred_list.append(3)
            if (t_y[i])==1:
                real_list.append(1)
            if (t_y[i])==2:
                real_list.append(2)
            if (t_y[i])==3:
                real_list.append(3) 
    print('Confusion Matrix:\n',confusion_matrix(real_list,pred_list))
#Function to calculate the confusion matrix for the wine dataset
def Confusion_matrix(t_x,t_y,tree):
    pred_list=[]
    real_list=[]
    for i in range(t_x.shape[0]):
        if predict(tree,t_x[i])=='positive':
            pred_list.append(1)
            if (t_y[i])=='positive':
                real_list.append(1)
            if (t_y[i])=='negative':
                real_list.append(0)
        if predict(tree,t_x[i])=='negative':
            pred_list.append(0)
            if (t_y[i])=='positive':
                real_list.append(1)
            if (t_y[i])=='negative':
                real_list.append(0)   
    print('Confusion Matrix:\n',confusion_matrix(real_list,pred_list))
#Function to calculate the entropy of the tic tac toe dataset
def entropy(y):
    all=y.size
    if all!=0:
        pos=np.where(y=='positive')[0]
        neg=np.where(y=='negative')[0]
        pos=len(pos)/all
        neg=len(neg)/all
        return -pos*np.log(pos+0.0001)-neg*np.log(neg+0.0001)
    if all==0:
        return -500
#FUNCTION TO RETURN THE INDEX AT WHICH THE DECSION TREE IS SPLIT FOR THE TIC TAC TOE DATA USING THE INFORMATION GAIN
def find_split_information_gain(x,y):
    all_features=x.shape[1]
    information_gain=[]
    root_entropy= entropy(y)
    for i in range(all_features):
        node_left_index= np.where(x[:,i]=='x')[0]
        node_left_data=y[node_left_index]
        node_mid_index= np.where(x[:,i]=='o')[0]
        node_mid_data=y[node_mid_index]
        node_right_index= np.where(x[:,i]=='b')[0]
        node_right_data=y[node_right_index]
        flagoraray_index=-500
        if len(node_left_index)*len(node_mid_index)*len(node_right_index)!=0:
            flagoraray_index= root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_mid_index)/len(y)*entropy(node_mid_data) - len(node_right_index)/len(y)*entropy(node_right_data)
        if len(node_left_index)==0 and len(node_right_index)*len(node_mid_index)!=0:    
            flagoraray_index= root_entropy - len(node_mid_index)/len(y)*entropy(node_mid_data) - len(node_right_index)/len(y)*entropy(node_right_data)
        if len(node_mid_index)==0  and len(node_right_index)*len(node_left_index)!=0:
            flagoraray_index= root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_right_index)/len(y)*entropy(node_right_data)   
        if len(node_right_index)==0  and len(node_mid_index)*len(node_left_index)!=0:
            flagoraray_index= root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_mid_index)/len(y)*entropy(node_mid_data) 
        information_gain.append(flagoraray_index)    
    information_gain=np.array(information_gain)
    split_index=np.argmax(information_gain)
    return split_index
#Class which represents the node for the decision tree for tic tac toe data
class node: 
    def __init__(self,x,y,leaf=False):
        self.data=x
        self.label=y
        self.is_leaf=leaf
        self.split_attribute=None
        self.classification=None
        self.node_entropy= entropy(self.label)
        if self.node_entropy>0:
            split_index=find_split_information_gain(self.data,self.label)
            self.split_attribute=split_index
            self.node_left_index= np.where(x[:,split_index]=='x')[0]
            self.node_left_y=y[self.node_left_index]
            self.node_left_x=x[self.node_left_index] 
            self.node_mid_index= np.where(x[:,split_index]=='o')[0]
            self.node_mid_y=y[self.node_mid_index]
            self.node_mid_x=x[self.node_mid_index]
            self.node_right_index= np.where(x[:,split_index]=='b')[0]
            self.node_right_y=y[self.node_right_index]
            self.node_right_x=x[self.node_right_index]
            self.node_left=node(self.node_left_x,self.node_left_y)
            self.node_mid=node(self.node_mid_x,self.node_mid_y)
            self.node_right=node(self.node_right_x,self.node_right_y) 
        else:
            if len(self.label)>0:
                self.is_leaf=True            
                u,f= np.unique(self.label,return_counts=True)  
                self.classification=u[np.argmax(f)]
#Function to predict the labels for the test datasets for tic tac dataset
def predict(tree,test):
    test_tree=tree
    while True:
        if test_tree.is_leaf==True or test_tree.split_attribute==None:
            return test_tree.classification
            break
        index=test_tree.split_attribute
        if test[index]=='x':
            test_tree=test_tree.node_left
        if test[index]=='o':
            test_tree=test_tree.node_mid
        if test[index]=='b':
            test_tree=test_tree.node_right
#Function to calculate accuracy using the predicted values for the tic tac toe dataset
def Accuracy(x_test,y_test,tree):
    count=0
    for i in range(x_test.shape[0]):
        if predict(tree,x_test[i])== y_test[i]:
            count=count+1
    return count
#Function which calculates the split index using the gain ratio for tic tac toe dataset
def find_split_gain_ratio(x,y):
    all_features=x.shape[1] 
    information_gain=[] 
    root_entropy= entropy(y)
    for i in range(all_features):
        node_left_index= np.where(x[:,i]=='x')[0]
        node_left_data=y[node_left_index]
        node_mid_index= np.where(x[:,i]=='o')[0]
        node_mid_data=y[node_mid_index]
        node_right_index= np.where(x[:,i]=='b')[0]
        node_right_data=y[node_right_index]
        flagoraray_index=-500
        if len(node_left_index)*len(node_mid_index)*len(node_right_index)!=0:
            flagoraray_index= (root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_mid_index)/len(y)*entropy(node_mid_data)-len(node_right_index)/len(y)*entropy(node_right_data))/ (- len(node_left_index)/len(y)*np.log(len(node_left_index)/len(y)) - len(node_mid_index)/len(y)*np.log(len(node_mid_index)/len(y)) - len(node_right_index)/len(y)*np.log(len(node_right_index)/len(y)))       
        if len(node_left_index)==0 and len(node_right_index)*len(node_mid_index)!=0:   
            flagoraray_index= (root_entropy   - len(node_mid_index)/len(y)*entropy(node_mid_data) - len(node_right_index)/len(y)*entropy(node_right_data))/(-len(node_mid_index)/len(y)*np.log(len(node_mid_index)/len(y)) - len(node_right_index)/len(y)*np.log(len(node_right_index)/len(y)))   
        if len(node_mid_index)==0  and len(node_right_index)*len(node_left_index)!=0:
            flagoraray_index= (root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_right_index)/len(y)*entropy(node_right_data))/(-len(node_left_index)/len(y)*np.log(len(node_left_index)/len(y)) - len(node_right_index)/len(y)*np.log(len(node_right_index)/len(y)))       
        if len(node_right_index)==0  and len(node_mid_index)*len(node_left_index)!=0:
            flagoraray_index= (root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_mid_index)/len(y)*entropy(node_mid_data))/(-len(node_left_index)/len(y)*np.log(len(node_left_index)/len(y)) - len(node_mid_index)/len(y)*np.log(len(node_mid_index)/len(y)))   
        information_gain.append(flagoraray_index)    
    information_gain=np.array(information_gain) 
    split_index=np.argmax(information_gain)
    return split_index
#Class which represents the node for the decision tree for tic tac toe data using gain ratio
class node_gr:
    def __init__(self,x,y,leaf=False):
        self.data=x
        self.label=y
        self.is_leaf=leaf
        self.split_attribute=None
        self.classification=None
        self.node_entropy= entropy(self.label)
        if self.node_entropy>0:
            split_index=find_split_gain_ratio(self.data,self.label)
            self.split_attribute=split_index
            self.node_left_index= np.where(x[:,split_index]=='x')[0]
            self.node_left_y=y[self.node_left_index]
            self.node_left_x=x[self.node_left_index]
            self.node_mid_index= np.where(x[:,split_index]=='o')[0]
            self.node_mid_y=y[self.node_mid_index]
            self.node_mid_x=x[self.node_mid_index]
            self.node_right_index= np.where(x[:,split_index]=='b')[0]
            self.node_right_y=y[self.node_right_index]
            self.node_right_x=x[self.node_right_index]
            self.node_left=node_gr(self.node_left_x,self.node_left_y)
            self.node_mid=node_gr(self.node_mid_x,self.node_mid_y)
            self.node_right=node_gr(self.node_right_x,self.node_right_y)
        else:
            if len(self.label)>0:
                self.is_leaf=True            
                u,f= np.unique(self.label,return_counts=True)  
                self.classification=u[np.argmax(f)]
#Function to calculate the entropy using the wine dataset
def wine_entropy(y):
    all=y.size
    if all!=0:
        pos1=np.where(y==1)[0]
        pos2=np.where(y==2)[0]
        pos3=np.where(y==3)[0]       
        pos1=len(pos1)/all
        pos2=len(pos2)/all
        pos3=len(pos3)/all
        return -pos1*np.log(pos1+0.0001)-pos2*np.log(pos2+0.0001) -pos3*np.log(pos3+0.0001)
    if all==0:
        return -500
#The function to calculate the index at which tree is split using information gain for the wine dataset
def wine_find_split(x,y):
    number_features=x.shape[1]
    information_gain=[]
    root_entropy= wine_entropy(y)
    for i in range(number_features):
        ent_j=[]
        best=[]
        thresh=np.linspace(x[:,i].min(),x[:,i].max(),num=(x[:,i].max()-x[:,i].min()+0.5)/0.01)
        for j in thresh:
            node_left_index= np.where(x[:,i]<=j)[0]
            node_left_data=y[node_left_index]
            node_right_index= np.where(x[:,i]>=j)[0]
            node_right_data=y[node_right_index]
            flag=-500
            if len(node_left_index)*len(node_right_index)!=0:
                flag= root_entropy - len(node_left_index)/len(y)*wine_entropy(node_left_data) - len(node_right_index)/len(y)*wine_entropy(node_right_data)
            if len(node_left_index)==0 and len(node_right_index)!=0:
                flag= root_entropy  - len(node_right_index)/len(y)*wine_entropy(node_right_data)
            if len(node_right_index)==0  and len(node_left_index)!=0:
                flag= root_entropy - len(node_left_index)/len(y)*wine_entropy(node_left_data)  
            ent_j.append(flag)
            best.append(j)
        best=np.array(best)
        ent_j=np.array(ent_j)
        j_index=np.argmax(ent_j)  
        j_splits=[ best[j_index] , ent_j[j_index]  ]
        information_gain.append(j_splits)       
    information_gain=np.array(information_gain)  
    split_index=np.argmax(information_gain[:,1]) 
    return split_index,information_gain[split_index,0]
#The function to calculate the index at which tree is split using the gain ratio for the wine dataset
def wine_find_split_gr(x,y):
    number_features=x.shape[1]   
    information_gain=[]   
    root_entropy= wine_entropy(y)    
    for i in range(number_features):
        ent_j=[]
        best=[]
        thresh=np.linspace(x[:,i].min(),x[:,i].max(),num=(x[:,i].max()-x[:,i].min()+0.5)/0.01)      
        for j in thresh:
            node_left_index= np.where(x[:,i]<=j)[0]
            node_left_data=y[node_left_index]
            node_right_index= np.where(x[:,i]>=j)[0]
            node_right_data=y[node_right_index]
            flag=-500
            if len(node_left_index)*len(node_right_index)!=0:
                flag= (root_entropy - len(node_left_index)/len(y)*wine_entropy(node_left_data) - len(node_right_index)/len(y)*wine_entropy(node_right_data) ) / (- len(node_left_index)/len(y)*np.log(len(node_left_index)/len(y)) - len(node_right_index)/len(y)*np.log(len(node_right_index)/len(y)) )      
            if len(node_left_index)==0 and len(node_right_index)!=0:
                flag= (root_entropy  - len(node_right_index)/len(y)*wine_entropy(node_right_data)) / (- len(node_right_index)/len(y)*np.log(len(node_right_index)/len(y)))     
            if len(node_right_index)==0  and len(node_left_index)!=0:
                flag= (root_entropy - len(node_left_index)/len(y)*wine_entropy(node_left_data) ) / (- len(node_left_index)/len(y)*np.log(len(node_left_index)/len(y)))          
            ent_j.append(flag)
            best.append(j)       
        best=np.array(best)
        ent_j=np.array(ent_j)        
        j_index=np.argmax(ent_j)     
        j_splits=[ best[j_index] , ent_j[j_index]   ]       
        information_gain.append(j_splits)       
    information_gain=np.array(information_gain)    
    split_index=np.argmax(information_gain[:,1])   
    return split_index,information_gain[split_index,0]
#Class which represents the node for the decision tree for wine data using gain ratio
class wine_node_gr:
    def __init__(self,x,y,leaf=False):       
        self.data=x
        self.label=y
        self.is_leaf=leaf
        self.split_attribute=None
        self.splitval=None 
        self.classification=None        
        self.node_entropy= wine_entropy(self.label)
        if self.node_entropy>0:            
            split_index,split_value=wine_find_split_gr(self.data,self.label)           
            self.split_attribute=split_index
            self.splitval=split_value            
            self.node_left_index= np.where(x[:,split_index]<=self.splitval)[0]
            self.node_left_y=y[self.node_left_index]
            self.node_left_x=x[self.node_left_index]        
            self.node_right_index= np.where(x[:,split_index]>=self.splitval)[0]
            self.node_right_y=y[self.node_right_index]
            self.node_right_x=x[self.node_right_index]           
            self.node_left=wine_node_gr(self.node_left_x,self.node_left_y)
            self.node_right=wine_node_gr(self.node_right_x,self.node_right_y)        
        else:
            if len(self.label)>0:
                self.is_leaf=True            
                u,f= np.unique(self.label,return_counts=True)  
                self.classification=u[np.argmax(f)]
#Class which represents the node for the decision tree for tic tac toe data using gain ratio
class wine_node:
    def __init__(self,x,y,leaf=False):  
        self.data=x
        self.label=y
        self.is_leaf=leaf
        self.split_attribute=None
        self.splitval=None 
        self.classification=None       
        self.node_entropy= wine_entropy(self.label)
        if self.node_entropy>0:            
            split_index,split_value=wine_find_split(self.data,self.label)            
            self.split_attribute=split_index
            self.splitval=split_value           
            self.node_left_index= np.where(x[:,split_index]<=self.splitval)[0]
            self.node_left_y=y[self.node_left_index]
            self.node_left_x=x[self.node_left_index]       
            self.node_right_index= np.where(x[:,split_index]>=self.splitval)[0]
            self.node_right_y=y[self.node_right_index]
            self.node_right_x=x[self.node_right_index]           
            self.node_left=wine_node(self.node_left_x,self.node_left_y)
            self.node_right=wine_node(self.node_right_x,self.node_right_y)       
        else:
            if len(self.label)>0:
                self.is_leaf=True            
                u,f= np.unique(self.label,return_counts=True)  
                self.classification=u[np.argmax(f)]
#Function to predict the labels for the test data
def wine_pred(tree,test): 
    test_tree=tree   
    while True:  
        if test_tree.is_leaf==True or test_tree.split_attribute==None:   
            return test_tree.classification
            break
        index=test_tree.split_attribute
        sp_val=test_tree.splitval
        if test[index]<sp_val:
            test_tree=test_tree.node_left
        if test[index]>=sp_val:
            test_tree=test_tree.node_right
#function to calculate the accuracy using predict function to calculate the acciuracy
def wine_acc(x_test,y_test,tree):
    count=0
    for i in range(x_test.shape[0]):
        if wine_pred(tree,x_test[i])== y_test[i]:
            count=count+1
    return count
# Main function to perform the 10 times 10 fold validation  for the wine and tic tac toe dataset using both information gain and gain ratio
def main():
    print("FOR TIC TAC TOE DATA") 
    #To calculate the 10 times 10 fold accuracy for tic tac toe data using the information Gain
    k=KFold(n_splits=10, random_state=None, shuffle=True)
    index= np.arange(957)
    fold=0
    accuracy=[]
    a=[]
    for i in range(10):
        for train_index, test_index in k.split(index):
            train_x=x[train_index]
            train_y=y[train_index]
            t_x=x[test_index]
            t_y=y[test_index]
            tree=node(train_x,train_y)
            flagoraray_index=Accuracy(t_x,t_y,tree)
            accuracy.append(flagoraray_index/t_y.size)     
        print("Accuracy with Information Gain for",i+1,"th time is",np.array(accuracy).mean())
        a.append(np.array(accuracy).mean())
    print("Accuracy with Information Gain for 10 iterations is", np.array(a).mean())  
    print("The standard deviation for Information Gain for 10 iterations is",np.std(np.array(a)))
    Confusion_matrix(t_x,t_y,tree)
    #To calculate the 10 times 10 fold accuracy for tic tak toe data using the Gain Ratio
    k=KFold(n_splits=10, random_state=None, shuffle=True)
    index=np.arange(957)
    fold=0
    accuracy=[]
    a=[]
    for i in range(10):
        for train_index, test_index in k.split(index):
            train_x=x[train_index]
            train_y=y[train_index]
            t_x=x[test_index]
            t_y=y[test_index]
            tree=node_gr(train_x,train_y)
            flagoraray_index=Accuracy(t_x,t_y,tree)
            accuracy.append(flagoraray_index/t_y.size)
        print("Accuracy with Gain Ratio for",i+1,"th time  is ",np.array(accuracy).mean())
        a.append(np.array(accuracy).mean())
    print("Accuracy with Gain Ratio for 10 iterations is", np.array(a).mean())
    print("The standard deviation for Gain Ratio for 10 iterations is",np.std(np.array(a)))
    Confusion_matrix(t_x,t_y,tree)
    print("FOR WINE DATA")
    #To calculate the 10 times 10 fold accuracy for wine data using the information gain
    k=KFold(n_splits=10, random_state=None, shuffle=True)
    index=np.arange(177)
    fold=0
    accuracy=[]
    a=[]
    for i in range(10):
        for train_index, test_index in k.split(index):       
            train_x=w_x[train_index]
            train_y=w_y[train_index]        
            t_x=w_x[test_index]
            t_y=w_y[test_index] 
            tree=wine_node(train_x,train_y)     
            flag=wine_acc(t_x,t_y,tree)
            accuracy.append(flag/t_y.size)
        print("Accuracy with Information gain for",i+1,"th time  is ",np.array(accuracy).mean())
        a.append(np.array(accuracy).mean())
    print("Accuracy with Information Gain for 10 iterations is", np.array(a).mean())
    print("The standard deviation for Information Gain for 10 iterations is",np.std(np.array(a)))
    wine_confusion_matrix(t_x,t_y,tree)
    #To calculate the 10 times 10 fold accuracy for wine data using the Gain Ratio
    k=KFold(n_splits=10, random_state=None, shuffle=True)
    index=np.arange(177)
    fold=0
    accuracy=[]
    a=[]
    for i in range(10):
        for train_index, test_index in k.split(index):       
            train_x=w_x[train_index]
            train_y=w_y[train_index]        
            t_x=w_x[test_index]
            t_y=w_y[test_index]       
            tree=wine_node_gr(train_x,train_y)        
            flag=wine_acc(t_x,t_y,tree)
            accuracy.append(flag/t_y.size)
        print("Accuracy with Gain Ratio for",i+1,"th time  is ",np.array(accuracy).mean())
        a.append(np.array(accuracy).mean())
    print("Accuracy with Gain Ratio for 10 iterations is", np.array(a).mean())
    print("The standard deviation for Gain Ratio for 10 iterations is",np.std(np.array(a)))
    wine_confusion_matrix(t_x,t_y,tree)
if __name__== "__main__":
  main()