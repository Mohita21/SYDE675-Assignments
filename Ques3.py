#Code for Question 3
#importing the necessary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#importing tic tac toe data using the link and separating the labels and features
link='https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'
d=pd.read_csv(link)
d=np.array(d)
d_y=d[:,9]
d_x=d[:,:9]
#importing the wine data using the link and separating the labels and features.
link='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine_data=pd.read_csv(link)
wine_data=np.array(wine_data)
wine_y=wine_data[:,0]
wine_x=wine_data[:,1:]
wine_y=wine_y.astype('int')
# Function to calculate the entropy for the wine data
def wine_entropy(y):
    count=y.size
    if count!=0:
        pos1=np.where(y==1)[0]
        pos2=np.where(y==2)[0]
        pos3=np.where(y==3)[0]        
        pos1=len(pos1)/count
        pos2=len(pos2)/count
        pos3=len(pos3)/count
        en=-pos1*np.log(pos1+0.0001)-pos2*np.log(pos2+0.0001) -pos3*np.log(pos3+0.0001)
        return en
    if count==0:
        return -500
# FUNCTION TO RETURN THE INDEX AT WHICH THE DECSION TREE IS SPLIT FOR THE TIC TAC TOE DATA USING THE INFORMATION GAIN   
def wine_Split(x,y):  
    number_features=x.shape[1]    
    information_gain=[]   
    root_entropy= wine_entropy(y) 
    for i in range(number_features):
        ent_j=[]
        j_b=[]
        thresh=np.linspace(x[:,i].min(),x[:,i].max(),num=(x[:,i].max()-x[:,i].min()+0.5)/0.01)        
        for j in thresh:
            node_left_index= np.where(x[:,i]<=j)[0]
            node_left_data=y[node_left_index]
            node_right_index= np.where(x[:,i]>=j)[0]
            node_right_data=y[node_right_index]
            flag=-500
            if len(node_left_index)*len(node_right_index)!=0:
                flag= root_entropy -len(node_left_index)/len(y)*wine_entropy(node_left_data)-len(node_right_index)/len(y)*wine_entropy(node_right_data)
            if len(node_left_index)==0 and len(node_right_index)!=0:
                flag= root_entropy - len(node_right_index)/len(y)*wine_entropy(node_right_data)
            if len(node_right_index)==0  and len(node_left_index)!=0:
                flag= root_entropy - len(node_left_index)/len(y)*wine_entropy(node_left_data)          
            ent_j.append(flag)
            j_b.append(j)        
        j_b=np.array(j_b)
        ent_j=np.array(ent_j)        
        j_index=np.argmax(ent_j)            
        j_splits=[j_b[j_index] , ent_j[j_index]]        
        information_gain.append(j_splits)        
    information_gain=np.array(information_gain)    
    split_index=np.argmax(information_gain[:,1])    
    return split_index,information_gain[split_index,0]
#Class which represents the node for the decision tree for wine data
class Wine_node:
    def __init__(self,x,y,depth,leaf=False):
        self.split_value=None 
        self.classification=None
        self.depth=depth 
        self.data=x
        self.label=y
        self.node_entropy= wine_entropy(self.label)
        self.is_leaf=leaf
        self.split_attribute=None
        if self.node_entropy>0 and self.depth<3:     
            split_index,split_value=wine_Split(self.data,self.label)  
            self.split_attribute=split_index
            self.split_value=split_value            
            self.node_left_index= np.where(x[:,split_index]<=self.split_value)[0]
            self.node_left_y=y[self.node_left_index]
            self.node_left_x=x[self.node_left_index]       
            self.node_right_index= np.where(x[:,split_index]>=self.split_value)[0]
            self.node_right_y=y[self.node_right_index]
            self.node_right_x=x[self.node_right_index]            
            self.node_left=Wine_node(self.node_left_x,self.node_left_y,self.depth+1)
            self.node_right=Wine_node(self.node_right_x,self.node_right_y,self.depth+1)        
        else:
            if len(self.label)>0:
                self.is_leaf=True            
                u,f= np.unique(self.label,return_counts=True)  
                self.classification=u[np.argmax(f)]
# function to predict labels for test data for wine
def Wine_predict(tree,test): 
    test_tree=tree
    while True:
        if test_tree.is_leaf==True or test_tree.split_attribute==None:       
            return test_tree.classification
            break
        index=test_tree.split_attribute
        sp_val=test_tree.split_value 
        if test[index]<sp_val:
            test_tree=test_tree.node_left
        if test[index]>=sp_val:
            test_tree=test_tree.node_right
# function top predict the accuracy for wine dataq usin g predict function
def Wine_accuracy(x_test,y_test,tree):
    count=0
    for i in range(x_test.shape[0]):
        if Wine_predict(tree,x_test[i])== y_test[i]:
            count=count+1
    return count
# Function to find the accuracy for clean train and clean test
def Wine_CleanTrainCleanTest():
    e_acc=[]
    for  e in range(10):
        k=KFold(n_splits=10, random_state=None, shuffle=True)
        index=np.arange(177)
        fold=0
        accuracy=[]
        for train_index, test_index in k.split(index):
            train_x=wine_x[train_index]
            train_y=wine_y[train_index]
            test_x=wine_x[test_index]
            test_y=wine_y[test_index]
            tree=Wine_node(train_x,train_y,0)
            flag=Wine_accuracy(test_x,test_y,tree)
            accuracy.append(flag/test_y.size)
        e_acc.append(np.array(accuracy).mean())
    mean_acc=np.array(e_acc).mean()
    std_acc=np.std(np.array(e_acc))
    print("\n For Clean Train and Clean Test")
    print("The mean accuracy is",mean_acc,"and The Standard Deviation is",std_acc)
    CvsC=np.array([mean_acc,std_acc])
    return CvsC
#Function to find the accuracy for Dirty train and Clean test (generating wrong attribute values for training data)
def Wine_DirtyTrainCleanTest():
    e_acc=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            k=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(177)
            fold=0
            accuracy=[]
            for train_index, test_index in k.split(index):
                train_x=wine_x[train_index]
                train_y=wine_y[train_index]
                no_c=int(np.floor(len(train_index)*l/100))
                index=np.arange(train_x.shape[0])
                np.random.shuffle(index)
                c_ind=index[0:no_c]
                for i in c_ind:
                    noise=np.random.normal(0,1,(1,13))
                    train_x[i]=train_x[i]+noise
                test_x=wine_x[test_index]
                test_y=wine_y[test_index]
                tree=Wine_node(train_x,train_y,1)
                flag=Wine_accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            e_acc.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    e_acc=np.array(e_acc)
    DvsC=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if e_acc[i]['L']==l:
                flag.append(e_acc[i]['acc'])
        flag=np.array(flag)
        print("\n For Dirty Train and Clean Test")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        DvsC.append([flag.mean(),np.std(flag)])
    DvsC=np.array(DvsC) 
    return DvsC
#function to generate the clean train and dirty test data and find accuracy for wine data
def Wine_CleanTrainDirtyTest():
    e_acc=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            from sklearn.model_selection import KFold
            kf=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(177)
            fold=0
            accuracy=[]
            for train_index, test_index in kf.split(index):
                train_x=wine_x[train_index]
                train_y=wine_y[train_index]
                test_x=wine_x[test_index]
                test_y=wine_y[test_index]
                no_c=int(np.floor(len(test_index)*l/100))
                index=np.arange(test_x.shape[0])
                np.random.shuffle(index)
                c_ind=index[0:no_c]
                for i in c_ind:
                    noise=np.random.normal(0,1,(1,13))
                    test_x[i]=test_x[i]+noise
                tree=Wine_node(train_x,train_y,1)
                flag=Wine_accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            e_acc.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    e_acc=np.array(e_acc)
    CvsD=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if e_acc[i]['L']==l:
                flag.append(e_acc[i]['acc'])
        flag=np.array(flag)
        print("\n For Clean Train and Dirty Test")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        CvsD.append([flag.mean(),np.std(flag)])
    CvsD=np.array(CvsD)
    return CvsD
#function to generate the dirty train and dirty test data and find accuracy for wine data
def Wine_DirtyTrainDirtyTest():
    e_acc=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            k=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(177)
            fold=0
            accuracy=[]
            for train_index, test_index in k.split(index):
                train_x=wine_x[train_index]
                train_y=wine_y[train_index]
                no_c=int(np.floor(len(train_index)*l/100))
                index=np.arange(train_x.shape[0])
                np.random.shuffle(index)
                c_ind=index[0:no_c]
                for i in c_ind:
                    noise=np.random.normal(0,1,(1,13))
                    train_x[i]=train_x[i]+noise
                test_x=wine_x[test_index]
                test_y=wine_y[test_index]
                no_c=int(np.floor(len(test_index)*l/100))
                index=np.arange(test_x.shape[0])
                np.random.shuffle(index)
                c_ind=index[0:no_c]
                for i in c_ind:
                    noise=np.random.normal(0,1,(1,13))
                    test_x[i]=test_x[i]+noise
                tree=Wine_node(train_x,train_y,1)
                flag=Wine_accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            e_acc.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    e_acc=np.array(e_acc)
    DvsD=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if e_acc[i]['L']==l:
                flag.append(e_acc[i]['acc'])
        flag=np.array(flag)
        print("\n For Dirty Train and Dirty Test")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        DvsD.append([flag.mean(),np.std(flag)])
    DvsD=np.array(DvsD)
    return DvsD
# function to generate contradictory labels and find accuracy for wine data
def Wine_Contradiction_labels():   
    e_acc=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            from sklearn.model_selection import KFold
            kf=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(177)
            fold=0
            accuracy=[]
            for train_index, test_index in kf.split(index):
                train_x=wine_x[train_index]
                train_y=wine_y[train_index]
                no_c=int(np.floor(len(train_index)*l/100))
                index=np.arange(train_x.shape[0])
                np.random.shuffle(index)
                c_ind=index[0:no_c]
                for i in c_ind:
                    train_x=np.concatenate([train_x,train_x[i].reshape(1,13)],axis=0)
                    contra_label=train_y[i]
                    while(contra_label==train_y[i]):
                        rand_index=np.random.randint(1,4)
                        contra_label=rand_index
                    train_y=np.concatenate([train_y,np.array(contra_label).reshape(1)])
                test_x=wine_x[test_index]
                test_y=wine_y[test_index]
                tree=Wine_node(train_x,train_y,1)
                flag=Wine_accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            e_acc.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    e_acc=np.array(e_acc)
    noise_contradictory=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if e_acc[i]['L']==l:
                flag.append(e_acc[i]['acc'])
        flag=np.array(flag)
        print("For Contradictory examples")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        noise_contradictory.append([flag.mean(),np.std(flag)])
    noise_contradictory=np.array(noise_contradictory)
    return noise_contradictory
#function to generate the missclassified data labels and find accuracy for wine data
def Wine_Misclass_labels():
    e_acc=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            k=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(177)
            fold=0
            accuracy=[]
            for train_index, test_index in k.split(index):
                train_x=wine_x[train_index]
                train_y=wine_y[train_index]
                no_c=int(np.floor(len(train_index)*l/100))
                index=np.arange(train_x.shape[0])
                np.random.shuffle(index)
                c_ind=index[0:no_c]
                for i in c_ind:
                    c_label=train_y[i]
                    while(c_label==train_y[i]):
                        rand_index=np.random.randint(1,4)
                        c_label=rand_index
                    train_y[i]==c_label
                test_x=wine_x[test_index]
                test_y=wine_y[test_index]
                tree=Wine_node(train_x,train_y,1)
                flag=Wine_accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            e_acc.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    e_acc=np.array(e_acc)
    noise_misclass=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if e_acc[i]['L']==l:
                flag.append(e_acc[i]['acc'])
        flag=np.array(flag)
        print("\n For the misclassified Labels")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        noise_misclass.append([flag.mean(),np.std(flag)])
    noise_misclass=np.array(noise_misclass)
    return noise_misclass
#function  to find entropy for tic tac data
def entropy(y):
    count=y.size
    if count!=0:
        pos1=np.where(y=='positive')[0]
        pos2=np.where(y=='negative')[0]
        pos1=len(pos1)/count
        pos2=len(pos2)/count
        en=-pos1*np.log(pos1+0.0001)-pos2*np.log(pos2+0.0001)
        return en
    if count==0:
        return -500
#Funtion to find the index at which the decision tree is split for tic tac data
def Split(x,y):
    Features=x.shape[1]
    information_gain=[]
    root_entropy= entropy(y)
    for i in range(Features):   
        node_left_index= np.where(x[:,i]=='x')[0]
        node_left_data=y[node_left_index]
        node_mid_index= np.where(x[:,i]=='o')[0]
        node_mid_data=y[node_mid_index]
        node_right_index= np.where(x[:,i]=='b')[0]
        node_right_data=y[node_right_index] 
        flag=-500
        if len(node_left_index)*len(node_mid_index)*len(node_right_index)!=0:
            flag= root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_mid_index)/len(y)*entropy(node_mid_data)  - len(node_right_index)/len(y)*entropy(node_right_data)
        if len(node_left_index)==0 and len(node_right_index)*len(node_mid_index)!=0:     
            flag= root_entropy - len(node_mid_index)/len(y)*entropy(node_mid_data) - len(node_right_index)/len(y)*entropy(node_right_data)
        if len(node_mid_index)==0  and len(node_right_index)*len(node_left_index)!=0:
            flag= root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_right_index)/len(y)*entropy(node_right_data)    
        if len(node_right_index)==0  and len(node_mid_index)*len(node_left_index)!=0:
            flag= root_entropy - len(node_left_index)/len(y)*entropy(node_left_data) - len(node_mid_index)/len(y)*entropy(node_mid_data)   
        information_gain.append(flag)    
    information_gain=np.array(information_gain)  
    split_index=np.argmax(information_gain)
    return split_index
#Class which represents the node for the decision tree for tic tac toe data
class node:   
    def __init__(self,x,y,depth,leaf=False):
        self.classification=None
        self.depth=depth
        self.data=x
        self.label=y
        self.is_leaf=leaf
        self.splt_attribute=None 
        self.node_entropy= entropy(self.label)  
        if self.node_entropy>0 and self.depth<25:
            split_index=Split(self.data,self.label)
            self.splt_attribute=split_index
            self.node_left_index= np.where(x[:,split_index]=='x')[0]
            self.node_left_y=y[self.node_left_index]
            self.node_left_x=x[self.node_left_index]
            self.node_mid_index= np.where(x[:,split_index]=='o')[0]
            self.node_mid_y=y[self.node_mid_index]
            self.node_mid_x=x[self.node_mid_index]
            self.node_right_index= np.where(x[:,split_index]=='b')[0]
            self.node_right_y=y[self.node_right_index]
            self.node_right_x=x[self.node_right_index]
            self.node_left=node(self.node_left_x,self.node_left_y,self.depth+1)
            self.node_mid=node(self.node_mid_x,self.node_mid_y,self.depth+1)
            self.node_right=node(self.node_right_x,self.node_right_y,self.depth+1)
        else:
            if len(self.label)>0:
                self.is_leaf=True            
                u,v= np.unique(self.label,return_counts=True)  
                self.classification=u[np.argmax(v)]
# function which predicts the labels for test data.
def Predict(tree,test):  
    tree_t=tree    
    while True:    
        if tree_t.is_leaf==True or tree_t.splt_attribute==None:           
            return tree_t.classification
            break
        index=tree_t.splt_attribute       
        if test[index]=='x':
            tree_t=tree_t.node_left
        if test[index]=='o':
            tree_t=tree_t.node_mid
        if test[index]=='b':
            tree_t=tree_t.node_right
#function which returns the accuracy using the predict function using tic tac toe   
def Accuracy(x_test,y_test,tree):
    count=0
    for i in range(x_test.shape[0]):
        if Predict(tree,x_test[i])== y_test[i]:
            count=count+1
    return count


def CleanTrainCleanTest():
    eected_accuracy=[]
    for  e in range(10):
        k=KFold(n_splits=10, random_state=None, shuffle=True)
        index=np.arange(957)
        fold=0
        accuracy=[]
        for train_index, test_index in k.split(index):
            x=d_x[train_index]
            y=d_y[train_index]
            test_x=d_x[test_index]
            test_y=d_y[test_index]
            tree=node(x,y,1)
            flag=Accuracy(test_x,test_y,tree)
            accuracy.append(flag/test_y.size)
        eected_accuracy.append(np.array(accuracy).mean())
    accuracy_mean=np.array(eected_accuracy).mean()
    std_acc=np.std(np.array(eected_accuracy))
    print("\n For Clean Train and Clean Test \n")
    print("The mean acciuracy is ",accuracy_mean,"The standard deviation is",std_acc)
    cVSc=np.array([accuracy_mean,std_acc])
    return cVSc

#function to generate the dirty train and clean test data and find accuracy for tic tac data
def DirtyTrainCleanTest():
    eected_accuracy=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            k=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(957)
            fold=0
            accuracy=[]
            for train_index, test_index in k.split(index):
                x=d_x[train_index]
                y=d_y[train_index]
                co_count=int(np.floor(len(train_index)*l/100))
                index=np.arange(x.shape[0])
                np.random.shuffle(index)
                index_co=index[0:co_count]
                for i in index_co:
                    for fea in range(x.shape[1]):
                        rand_value=np.random.uniform(0,1)
                        if rand_value<0.33:
                            x[i,fea]='x'
                        if rand_value>=0.33 and rand_value<0.66:
                            x[i,fea]='o'
                        if rand_value>=0.66:
                            x[i,fea]='b'
                test_x=d_x[test_index]
                test_y=d_y[test_index]
                tree=node(x,y,1)
                flag=Accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            eected_accuracy.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    dVSc=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if eected_accuracy[i]['L']==l:
                flag.append(eected_accuracy[i]['acc'])
        flag=np.array(flag)
        print("\n For Dirty Train and Clean Test \n")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        dVSc.append([flag.mean(),np.std(flag)])
    dVSc=np.array(dVSc) 
    return dVSc

#function to generate the clean train and dirty test data and find accuracy for tic tac data
def CleanTrainDirtyTest():
    eected_accuracy=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            k=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(957)
            fold=0
            accuracy=[]
            for train_index, test_index in k.split(index):
                x=d_x[train_index]
                y=d_y[train_index]
                test_x=d_x[test_index]
                test_y=d_y[test_index]
                co_count=int(np.floor(len(test_index)*l/100))
                index=np.arange(test_x.shape[0])
                np.random.shuffle(index)
                index_co=index[0:co_count]
                for i in index_co:
                    for fea in range(test_x.shape[1]):
                        rand_value=np.random.uniform(0,1)
                        if rand_value<0.33:
                            test_x[i,fea]='x'
                        if rand_value>=0.33 and rand_value<0.66:
                            test_x[i,fea]='o'
                        if rand_value>=0.66:
                            test_x[i,fea]='b'
                tree=node(x,y,1)
                flag=Accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            eected_accuracy.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    eected_accuracy=np.array(eected_accuracy)
    cVSd=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if eected_accuracy[i]['L']==l:
                flag.append(eected_accuracy[i]['acc'])
        flag=np.array(flag)
        print("\n For Clean Train and Dirty Test \n")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        cVSd.append([flag.mean(),np.std(flag)])
    cVSd=np.array(cVSd)
    return cVSd
#function to generate the dirty train and dirty test data and find accuracy for tic tac data
def DirtyTrainDirtyTest():
    eected_accuracy=[]
    for  e in range(10):
        L=[5,10,15]
        for l in L:
            k=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(957)
            fold=0
            accuracy=[]
            for train_index, test_index in k.split(index):
                x=d_x[train_index]
                y=d_y[train_index]
                co_count=int(np.floor(len(train_index)*l/100))
                index=np.arange(x.shape[0])
                np.random.shuffle(index)
                index_co=index[0:co_count]
                for i in index_co:
                    for fea in range(x.shape[1]):
                        rand_value=np.random.uniform(0,1)
                        if rand_value<0.33:
                            x[i,fea]='x'
                        if rand_value>=0.33 and rand_value<0.66:
                            x[i,fea]='o'
                        if rand_value>=0.66:
                            x[i,fea]='b'      
                test_x=d_x[test_index]
                test_y=d_y[test_index]
                co_count=int(np.floor(len(test_index)*l/100))
                index=np.arange(test_x.shape[0])
                np.random.shuffle(index)
                index_co=index[0:co_count]
                for i in index_co:
                    for fea in range(test_x.shape[1]):
                        rand_value=np.random.uniform(0,1)
                        if rand_value<0.33:
                            test_x[i,fea]='x'
                        if rand_value>=0.33 and rand_value<0.66:
                            test_x[i,fea]='o'
                        if rand_value>=0.66:
                            test_x[i,fea]='b'
                tree=node(x,y,1)
                flag=Accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            eected_accuracy.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    eected_accuracy=np.array(eected_accuracy)
    dVsd=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if eected_accuracy[i]['L']==l:
                flag.append(eected_accuracy[i]['acc'])
        flag=np.array(flag)
        print("\n For Dirty Train and Dirty Test ")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        dVsd.append([flag.mean(),np.std(flag)])
    dVsd=np.array(dVsd)
    return dVsd
# function to generate contradictory labels and find accuracy for tic tac data
def Contradiction_labels():
    eected_accuracy=[]
    for e in range(10):
        L=[5,10,15]
        for l in L:
            kf=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(957)
            fold=0
            accuracy=[]
            for train_index, test_index in kf.split(index):
                x=d_x[train_index]
                y=d_y[train_index]
                co_count=int(np.floor(len(train_index)*l/100))
                index=np.arange(x.shape[0])
                np.random.shuffle(index)
                index_co=index[0:co_count]
                for i in index_co:
                    x=np.concatenate([x,x[i].reshape(1,9)],axis=0)
                    if y[i]=='positive':
                        y=np.concatenate([y,np.array('negative').reshape(1)])
                    if y[i]=='negative':
                        y=np.concatenate([y,np.array('positive').reshape(1)])
                test_x=d_x[test_index]
                test_y=d_y[test_index]
                tree=node(x,y,1)
                flag=Accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            eected_accuracy.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    eected_accuracy=np.array(eected_accuracy)
    noise_contradictory=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if eected_accuracy[i]['L']==l:
                flag.append(eected_accuracy[i]['acc'])
        flag=np.array(flag)
        print("\n For Contradictory examples")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        noise_contradictory.append([flag.mean(),np.std(flag)])
    noise_contradictory=np.array(noise_contradictory)
    return noise_contradictory
#function to generate the missclassified data labels and find accuracy for tic tac data
def Misclass_labels():
    eected_accuracy=[]
    for  e in range(10):
        L=[5,10,15]
        for l in L:
            from sklearn.model_selection import KFold
            kf=KFold(n_splits=10, random_state=None, shuffle=True)
            index=np.arange(957)
            fold=0
            accuracy=[]
            for train_index, test_index in kf.split(index):
                x=d_x[train_index]
                y=d_y[train_index]
                co_count=int(np.floor(len(train_index)*l/100))
                index=np.arange(x.shape[0])
                np.random.shuffle(index)
                index_co=index[0:co_count]
                for i in index_co:
                    if y[i]=='positive':
                        y[i]='negative'
                    if y[i]=='negative':
                        y[i]='positive'
                test_x=d_x[test_index]
                test_y=d_y[test_index]
                tree=node(x,y,1)
                flag=Accuracy(test_x,test_y,tree)
                accuracy.append(flag/test_y.size)
            eected_accuracy.append({'e':e,'L':l,'acc':np.array(accuracy).mean()})
    eected_accuracy=np.array(eected_accuracy)
    noise_mc=[]
    L=[5,10,15]
    for l in L:
        flag=[]
        for i in range(30):
            if eected_accuracy[i]['L']==l:
                flag.append(eected_accuracy[i]['acc'])
        flag=np.array(flag)
        print("\n For the misclassified labels")
        print('For noise level ',l,'% the mean accuracy is', flag.mean(),'and the Standard Deviation is',np.std(flag))
        noise_mc.append([flag.mean(),np.std(flag)])
    noise_mc=np.array(noise_mc)
    return  noise_mc
#Main function which calls the above functions to find the accuracy for CvC DvD CvD DvC and contradictory and misclassified data for tic tac toe and wine dataset
def main():
    # to calculate and plot 'Clean Vs Clean','Clean Vs Dirty','Dirty Vs Clean','Dirty Vs Dirty' for tic tac toe data
    cVSc=CleanTrainCleanTest()
    dVSc=DirtyTrainCleanTest()
    cVSd=CleanTrainDirtyTest()
    dVsd= DirtyTrainDirtyTest()
    L=[5,10,15]
    plt.plot(L,[cVSc[0],cVSc[0],cVSc[0]])
    plt.plot(L,cVSd[:,0])
    plt.plot(L,dVSc[:,0])
    plt.plot(L,dVsd[:,0])
    plt.legend(['Clean Vs Clean','Clean Vs Dirty','Dirty Vs Clean','Dirty Vs Dirty'])
    plt.show()
    # to calculate and plot 'Clean Vs Clean','Contradictory examples','Miss-Classfication'for tic tac toe data
    noise_contradictory=Contradiction_labels()
    noise_mc=Misclass_labels()
    plt.plot(L,[cVSc[0],cVSc[0],cVSc[0]])
    plt.plot(L,noise_contradictory[:,0])
    plt.plot(L,noise_mc[:,0])
    plt.legend(['Clean Vs Clean','Contradictory examples','Miss-Classfication'])
    plt.show()
    # to calculate and plot 'Clean Vs Clean','Clean Vs Dirty','Dirty Vs Clean','Dirty Vs Dirty' for wine data
    L=[5,10,15]
    CvsC=Wine_CleanTrainCleanTest()
    DvsC=Wine_DirtyTrainCleanTest()
    CvsD=Wine_CleanTrainDirtyTest()
    DvsD=Wine_DirtyTrainDirtyTest()
    plt.plot(L,[CvsC[0],CvsC[0],CvsC[0]])
    plt.plot(L,CvsD[:,0])
    plt.plot(L,DvsC[:,0])
    plt.plot(L,DvsD[:,0])
    plt.legend(['Clean Vs Clean','Clean Vs Dirty','Dirty Vs Clean','Dirty Vs Dirty'])
    plt.show()
    # to calculate and plot 'Clean Vs Clean','Contradictory examples','Miss-Classfication'for wine data
    noise_contradictory=Wine_Contradiction_labels()
    noise_misclass=Wine_Misclass_labels()
    plt.plot(L,[CvsC[0],CvsC[0],CvsC[0]])
    plt.plot(L,noise_contradictory[:,0])
    plt.plot(L,noise_misclass[:,0])
    plt.legend(['Clean Vs Clean','Contradictory examples','Miss-Classfication'])
    plt.show()

if __name__== "__main__":
  main()