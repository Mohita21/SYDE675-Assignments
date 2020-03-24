#Code for question 2 (Mohita Chaudhary)
#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from matplotlib.colors import ListedColormap
import matplotlib.patches as pat

#function to return sorted eigenvalues and eigenvectors
def Eig_sort(cov):
        val, vec = np.linalg.eigh(cov)
        order = val.argsort()[::-1]
        return val[order], vec[:,order]

#function to create a Maximum A posteriori Classifier
def MAP(values):
    l = list()
    #Storing the covariances
    A= np.array([[1, -1],[-1, 2]])
    B = np.array([[1, -1],[-1, 2]])
    C = np.array([[0.5, 0.5],[0.5, 3]])
    #Storing the means
    u1= np.array([3,2])
    u2= np.array([5,4])
    u3= np.array([2,5])
    #Storing the prior probabilities
    prA = 0.2
    prB = 0.3
    prC = 0.5
    #Obtaining the scores for MAP using the formula taught in class
    for x in values:
        s1 = (-0.5*np.dot(np.dot((x - u1), inv(A)),(x - u1).T) - 0.5 * np.log(norm(A)) + np.log(prA))
        s2 = (-0.5*np.dot(np.dot((x - u2), inv(B)),(x - u2).T) - 0.5 * np.log(norm(B)) + np.log(prB))
        s3 = (-0.5*np.dot(np.dot((x - u3), inv(C)),(x - u3).T) - 0.5 * np.log(norm(C)) + np.log(prC))
    #concatining the scores and finding the maximum score
        score = [s1, s2, s3]
        l.append(np.argmax(score))
    return l

#function to create a Maximum Likelihood Classifier
def ML(values):
    l = list()
    #Storing the given covariances
    A= np.array([[1, -1],[-1, 2]])
    B = np.array([[1, -1],[-1, 2]])
    C = np.array([[0.5, 0.5],[0.5, 3]])
    #Storing the mean of the classes
    u1= np.array([3,2])
    u2= np.array([5,4])
    u3= np.array([2,5])
    #Obtaining the scores for ML using the formula taught in class
    for x in values:
        s1 =(-0.5*np.dot(np.dot((x - u1),inv(A)),(x - u1).T) - 0.5 * np.log(norm(A)))
        s2 =(-0.5*np.dot(np.dot((x - u2),inv(B)),(x - u2).T) - 0.5 * np.log(norm(B)))
        s3 =(-0.5*np.dot(np.dot((x - u3),inv(C)),(x - u3).T) - 0.5 * np.log(norm(C)))
    #concatining the scores and finding the maximum score
        score = [s1, s2, s3]
        l.append(np.argmax(score))
    return l

#Function to plot the MAP and ML decision boundaries
def DecisionBoundaries():
    A= np.array([[1, -1],[-1, 2]])
    B = np.array([[1, -1],[-1, 2]])
    C = np.array([[0.5, 0.5],[0.5, 3]])
    
    #Using the defined function to store the eigenvalues and eigenvectors for each class in sorted order
    A_value, A_vector = Eig_sort(A)
    B_value, B_vector = Eig_sort(B)
    C_value, C_vector =  Eig_sort(C)
    #Storing the angle for each contour for each class
    thetaA = np.degrees(np.arctan2(*A_vector[:,0][::-1]))
    thetaB = np.degrees(np.arctan2(*B_vector[:,0][::-1]))
    thetaC = np.degrees(np.arctan2(*C_vector[:,0][::-1]))
    #Storing the height and width for the contours for each class
    wA, hA = 2 * np.sqrt(A_value)
    wB, hB = 2 * np.sqrt(B_value)
    wC, hC = 2 * np.sqrt(C_value) 
    #defining the colors for decision boundaries
    colors = ListedColormap(['yellow', 'pink', 'purple'])
    labels = ['A' ,'B' ,'C']
    #using meshgrid to plot the contours
    X, Y = np.meshgrid(np.arange(-15, 15, 0.05),np.arange(-15, 15, 0.05))
    #stroing the results of classifiers in 1D vectors
    MAP_out= MAP(np.c_[X.ravel(), Y.ravel()])
    ML_out = ML(np.c_[X.ravel(), Y.ravel()])
    #Reshaping the results so that they can be used in plt.contour method
    MAP_out = np.reshape(MAP_out,X.shape)
    ML_out = np.reshape(ML_out,X.shape)
    #storing the mean as centers
    centerA=(3,2)
    centerB=(5,4)
    centerC=(2,5)
    #Using Ellipse method to plot the contours
    ax = plt.subplot(121,aspect='equal')
    contourA = pat.Ellipse(centerA,width=wA,height=hA,angle=thetaA)
    contourB = pat.Ellipse(centerB,width=wB,height=hB,angle=thetaB)
    contourC = pat.Ellipse(centerC,width=wC,height=hC,angle=thetaC)
    contourA.set_alpha(0.5)
    contourA.set_facecolor('black') 
    ax.add_artist(contourA)
    contourB.set_alpha(0.5)
    contourB.set_facecolor('black') 
    ax.add_artist(contourB)
    contourC.set_alpha(0.5)
    contourC.set_facecolor('black') 
    ax.add_artist(contourC)
    #plotting the colored data boundaries
    plt.pcolormesh(X, Y, MAP_out, cmap=colors)
    plt.contour(X, Y, MAP_out, colors='black')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("The Decision Boundary for MAP")
    
    #Repeating the same for ML decision boundaries.
    ax = plt.subplot(122, aspect='equal')
    contourA = pat.Ellipse(centerA,width=wA,height=hA,angle=thetaA)
    contourB = pat.Ellipse(centerB,width=wB,height=hB,angle=thetaB)
    contourC = pat.Ellipse(centerC,width=wC,height=hC,angle=thetaC)
    contourA.set_alpha(0.5)
    contourA.set_facecolor('black') 
    contourA.set_label('A')
    ax.add_artist(contourA)
    contourB.set_alpha(0.5)
    contourB.set_facecolor('black') 
    contourA.set_label('B')
    ax.add_artist(contourB)
    contourC.set_alpha(0.5)
    contourC.set_facecolor('black') 
    contourA.set_label('C')
    ax.add_artist(contourC)
    #plotting the colored data boundaries
    plt.pcolormesh(X, Y, ML_out, cmap=colors,label=labels)
    plt.contour(X, Y , ML_out, colors='black')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("The Decision Boundary for ML")
    plt.show()
    return

#function for calculating the confusion matrix
def confusion_matrix(lab,pred):
    con=np.zeros((3,3))
    pred=np.array(pred)
    for  i in range(3):
        true_i=np.where(lab+1==i+1)
        true_i=true_i[0]
        pred_i1=np.where(pred[true_i]+1==1)
        pred_i2=np.where(pred[true_i]+1==2)
        pred_i3=np.where(pred[true_i]+1==3) 
        con[i,:]=[len(pred_i1[0]),len(pred_i2[0]),len(pred_i3[0])]   
    return con
#main function to call the above defined functions
def main():
    DecisionBoundaries()
    #Generating the 3000 Samples
    classA = np.random.multivariate_normal([3,2], [[1, -1],[-1, 2]], 600)
    classB = np.random.multivariate_normal([5,4], [[1, -1],[-1, 2]], 2100) 
    classC = np.random.multivariate_normal([2,5], [[0.5, 0.5],[0.5, 3]], 300)
    #Explicitly generating the class labels using the priors 
    label_A = [0] * 600  #prA = 0.2 (0.2*3000=600)
    label_B = [1] * 2100  #prB = 0.7
    label_C = [2] * 300 #prC = 0.1
    #Predicted labels in a list
    labels = np.concatenate((label_A, label_B, label_C), axis=0)
    #Combining all the 3000 data into one list
    x =np.concatenate((classA, classB, classC), axis=0)
    #Calling MAP and ML classifiers
    map_res = MAP(x)
    ml_res = ML(x)
    #Calling the above defined function to calculate the Confusion matrix
    print('The Confusion Matrix for MAP Classifier-')
    conf_MAP=confusion_matrix(labels, map_res)
    print(conf_MAP)
    print('The Confusion Matrix for ML Classifier-')
    conf_ML=confusion_matrix(labels, ml_res)
    print(conf_ML)
    #The classification error for classes A, B and C
    print('\n The value of P(Error) for the MAP Classifier for A, B and C are as follows :\n A: {}, B: {}, C: {}'.format((conf_MAP[0][1]+conf_MAP[0][2])/3000,(conf_MAP[1][0]+conf_MAP[1][2])/3000,(conf_MAP[2][0]+conf_MAP[2][1])/3000))
    print('\n The value for P(Error) for the ML Classifier for A, B and C are as follows :\n A: {}, B: {}, C: {}'.format((conf_ML[0][1]+conf_ML[0][2])/3000,(conf_ML[1][0]+conf_ML[1][2])/3000,(conf_ML[2][0]+conf_ML[2][1])/3000))
    

if __name__== "__main__":
  main()