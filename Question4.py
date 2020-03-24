# Code for Q4 (Mohita Chaudhary)
# importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
#function to read the given dataset
def Read():
    f1=[]
    f2=[]
    labels=[]
    with open("/Users/mohita/Downloads/dataset3.txt","r") as files:
        for x in files:
            line = x.split(",")
            f1.append(line[0])
            f2.append(line[1])
            labels.append(line[2][0])
    #converting the features and labels to array
    f1=np.array(f1)
    f2=np.array(f2)
    labels=np.array(labels)
    #reshaping the arrays
    f1=f1.reshape(1,100)
    f2=f2.reshape(1,100)
    #converting the features and labels to float
    f1=f1.astype(np.float)
    f2=f2.astype(np.float)
    labels=labels.astype(np.float)
    f1=np.asarray(f1)
    f2=np.asarray(f2)
    #performing feature scaling
    f1=f1/np.max(f1)
    f1=f1-np.mean(f1)
    f2=f2/np.max(f2)
    f2=f2-np.mean(f2)
    f1=f1[0].tolist()
    f2=f2[0].tolist()
    #returning the labels and f1 and f2 in a correct format
    return f1,f2,labels


#Function to calculate h_theta
def Hypothesis(theta,X):
    h= 1/(1+np.exp(-np.matmul(theta.T,X)))
    return h
#function to calculate the cost function
def Cost_Function(theta,f1,f2,labels):
    c_fn=0
    bias=1
    for i in range(100):
        X=[ bias,f1[i],f2[i]]
        c_fn=c_fn + labels[i]*np.log(Hypothesis(theta,X)) + (1-labels[i])*np.log(1-Hypothesis(theta,X))
    j=-c_fn/100
    return j
#Function to estimate weights using the Stochastic Gradient Descent
def SGD(theta,index,f1,f2,labels):
    #taking the value of learning rate as 0.1
    alpha=0.1
    #taking the batch size as 10
    batch=10
    bias=1
    gradient =np.array([0,0,0])
    for i in range(len(index)):
        X=[1, f1[index[i]],f2[index[i]]]
        p1=( Hypothesis(theta,X) - labels[index[i]]) * bias
        p2=( Hypothesis(theta,X) - labels[index[i]]) * f1[index[i]]
        p3=( Hypothesis(theta,X) - labels[index[i]]) * f2[index[i]] 
        gradient= gradient + [ p1, p2 , p3 ]
    theta = theta - alpha * gradient/batch
    return theta
#Function to perform the testing and calculating the accuracy
def Testing(theta,f1,f2,labels):
    bias=1
    correct=0
    predicted_labels=[]
    for i in range(100):
        X=[bias,f1[i],f2[i]]
        if Hypothesis(theta,X)>0.5 and labels[i]==1:
            predicted_labels.append(1)
            correct=correct+1
        if Hypothesis(theta,X)<0.5 and labels[i]==0:
            predicted_labels.append(0)
            correct=correct+1
    return correct, predicted_labels
#To initialise the weights randomly
def Weight_Initialization():
    theta=[np.random.normal(0,0.01),np.random.normal(0,0.01),np.random.normal(0,0.01)]
    theta=np.array(theta)
    return theta
#To perform training by calling the above defined functions
def Training(steps,f1,f2,batch,labels):
    loss=[]
    #initialising the weight by callin g the above defined function
    theta=Weight_Initialization()
    for i in range(steps):
        k=np.arange(100)
        #shuffling the indexes
        np.random.shuffle(k)
        for j in range(int(100/batch)):
            q=j*10
            index=k[q:q+10]
            theta=SGD(theta,index,f1,f2,labels)
        #appending the loss obtained by calling the cost function defined above
        loss.append(Cost_Function(theta,f1,f2,labels))
        correct, predicted_labels= Testing(theta,f1,f2,labels)
        print("The accuracy for step ",i," is ",correct)
    return theta,loss
#function to plot the samples of class and to obtain the deciszion boundary
def plot_class_samples(f1,f2,theta,labels):
    plt.scatter(f1,f2,c=labels, marker='x',cmap='viridis')
    f_a=np.linspace(-0.6,0.6,100)
    f_b=(-theta[0]-theta[1]*f_a)/theta[2]
    plt.plot(f_a,f_b)
    plt.title("Samples for the two classes with decision boundary")
    plt.show()
    f1=np.linspace(-2,2,100)
    f2=(-theta[0]-theta[1]*f1)/theta[2]
    plt.plot(f1,f2)
    plt.title("The Decision Boundary")
    plt.show()
#Main function to call the above defined functions
def main():
    batch=10
    #reading thw labels and features in  a normalised form
    f1,f2,labels=Read()
    #calling the training function to perform training
    theta,loss=Training(2500,f1,f2,batch,labels)
    print("\n The estimated parameters are ",theta)
    plt.plot(loss)
    plt.title('Cost Function along the epochs of SGD')
    plt.show()
    plot_class_samples(f1,f2,theta,labels)
    accuracy,predicted_labels=Testing(theta,f1,f2,labels)
    print("\n The accuracy is ",accuracy)
    print("\n The predicted labels are ",predicted_labels)


if __name__== "__main__":
    main()

