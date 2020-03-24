#Code for Question 1 (Mohita Chaudhary)
#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
#Function to generate 1000 points for class A
def generate_1000_A():
    l=[]#list to store the random points
    x=[]#list to store the x coordinate
    y=[]#list to store the y coordinate
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    mean = np.array(mean)
    cov = np.array(cov)
    for i in range(1000):
        #numpy function to generate random numbers
        l.append(np.random.standard_normal(mean.size))
    for j in range(1000):
        m=l[j].tolist()
        x.append(m[0])
        y.append(m[1])
    return x,y
#Function to generate 1000 points for class B
def generate_1000_B():
    l=[]#list to store the random points
    x=[]#list to store the x coordinate
    y=[]#list to store the y coordinate
    mean = [0, 0]
    cov = [[2, -2], [-2, 3]]
    mean = np.array(mean)
    cov = np.array(cov)
    def eigh(mean, cov): 
        #inbuilt numpy function to calculate eigenvalue and eigenvectors
        val, v = np.linalg.eigh(cov) 
        #returning the randomly generated points transformed by the eigenvalues and eigenvectors for the covariance matrix
        return mean + v * np.sqrt(val) @ np.random.standard_normal(mean.size)
    for i in range(1000):
        #appending the 1000 random points in the list by calling the eigh function 1000 times.
        l.append(eigh(mean, cov))
    for j in range(1000):
        m=l[j].tolist()
        x.append(m[0])
        y.append(m[1])
    return x,y
#function which returns the equation of a circle
def circ_eqn(x, y):
    return x**2 + y**2-1

B=[[2, -2], [-2, 3]]
e_val,e_vec=np.linalg.eig(B)
a11=e_vec[0,0]
a12=e_vec[1,0]
a21=e_vec[0,1]
a22=e_vec[1,1]
val_1,val_2=e_val

#function returning the equation of the ellipse
def ellipse_eqn(x1, x2):
    return val_2*((a11*x1+a12*x2)**2) + val_1*((a21*x1+ a22*x2)**2) - 1

#function to generate the first standard deviation contour for the Class A and Class B
def contours():
    #Contour for Class A
    A = [[1, 0], [0, 1]]
    #statement to vectorize the equation of the circle.
    vector_function = np.vectorize(circ_eqn)
    #plotting the randomly generated points as well as the first std deviation contour
    x1, x2 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))#meshgrid used to generate the grid space.
    #calling the above defined function to store the random points for class A
    x, y = generate_1000_A()
    fig, ax = plt.subplots(1)
    plt.scatter(x,y,marker='x', s=20,c='red')
    ax.contour(x1, x2, vector_function(x1, x2),[0],colors='black')
    plt.show()
    
    #Contour for class B
    B=[[2, -2], [-2, 3]]
    #statement to vectorize the equation for contour for class B
    vector_function = np.vectorize(ellipse_eqn)
    x1, x2 = np.meshgrid(np.linspace(-5, 5, 100),np.linspace(-5, 5, 100))
    #calling the above defined function to store the random points for class B
    x, y = generate_1000_B()
    fig, ax = plt.subplots(1)
    plt.scatter(x,y,marker='x', s=20,c='green')
    ax.contour(x1, x2, vector_function(x1, x2),[0],colors='black')

    plt.show()
   
#Function to calcualte the covriance of the generated points
def covariance():
    #Calculating the covariance for the random points for Class A
    l=[]
    x=[]
    y=[]
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    mean = np.array(mean)
    cov = np.array(cov)
    for i in range(1000):
        l.append(np.random.standard_normal(mean.size))
    for j in range(1000):
        m=l[j].tolist()
        x.append(m[0])
        y.append(m[1])
    l_a=[]
    l_a.append(x)
    l_a.append(y)
    l_a=np.asarray(l_a)
    meanA = l_a.T - [l_a.T[:,0].mean(), l_a.T[:,1].mean()]
    #Storing the covariance in covA by using the following formula
    covA = np.matmul(meanA.T,meanA)/(1000-1)
    print("The sample covariance matrix for class A using the data generated is ",covA)
    #Calculating the covariance for the random points for Class A
    l=[]
    x=[]
    y=[]
    xB=[]
    mean = [0, 0]
    cov = [[2, -2], [-2, 3]]
    mean = np.array(mean)
    cov = np.array(cov)
    def eigh(mean, cov): 
        val, v = np.linalg.eigh(cov) 
        return mean + v * np.sqrt(val) @ np.random.standard_normal(mean.size)
    for i in range(1000):
        l.append(eigh(mean, cov))
    for j in range(1000):
        m=l[j].tolist()
        x.append(m[0])
        y.append(m[1])
    l_b=[]
    l_b.append(x)
    l_b.append(y)
    l_b=np.asarray(l_b)
    meanB = l_b.T - [l_b.T[:,0].mean(), l_b.T[:,1].mean()]
    #Storing the covariance in covB by using the following formula
    covB = np.matmul(meanB.T,meanB)/(1000-1)
    print("The sample covariance matrix for class B using the data generated is ",covB)

#Main function to call the above defined functions
def main():
    x,y=generate_1000_A()
    plt.scatter(x,y,marker='x', s=20,c='red')
    plt.axis('equal')
    plt.show()
    x,y=generate_1000_B()
    plt.scatter(x,y,marker='x', s=20,c='green')
    plt.axis('equal')
    plt.show()
    contours()
    covariance()
    
if __name__== "__main__":
  main()