#code for Q3(Mohita Chaudhary)
#importing all necessary libraries
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
#reading the data
x, y = loadlocal_mnist(
        images_path='/Users/mohita/Downloads/train-images-idx3-ubyte', 
        labels_path='/Users/mohita/Downloads/train-labels-idx1-ubyte')
x=x/255
x_t=x.T
#Function to compress the data using the PCA
def Principal_Component_Analysis(d,x):
    val,vec=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(np.abs(val))[::-1]
    vec_sorted=vec[indexes]
    first_d_vec=vec_sorted[:,0:d]
    transformed_data=np.matmul(first_d_vec.T,x.T)
    return transformed_data
#function to find the value of d for POV=95%
def Proportion_of_Variance(x):
    val,vec=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(np.abs(val))[::-1]
    val_sorted=val[indexes]
    val_sum=val_sorted.sum()
    for k in range(784):
        k_val_sum=val_sorted[:k+1].sum()
        POV=k_val_sum/val_sum
        if POV >= 0.95:
            break
    return(k+1)
#Function to reconstruct the compressed data
def Reconstruction(comp_x):
    reduced_dim=comp_x.shape[0]
    val,vec=np.linalg.eig(np.cov(x_t))
    indexes=np.argsort(np.abs(val))[::-1]
    vec_sorted=vec[indexes]
    first_d_vec=vec_sorted[:,0:reduced_dim]
    reconstructed_data=np.matmul(first_d_vec,comp_x)
    return reconstructed_data
#Function to calculate the MSE using the error between the reconstructed and original data
def Mean_Squared_Error(x):
    d=[]
    mse=[]
    #Appending the MSE for d=1
    d.append(1)
    d_pca=Principal_Component_Analysis(1,x)
    reconstructed_data=Reconstruction(d_pca)
    MSE=np.sum(np.sum(np.square(x_t-reconstructed_data),axis=0))/x_t.shape[1]
    mse.append(MSE)
    #appending the mse for the values between 1 and 784
    for k in range(1,40):
        l=20*k
        d.append(l)
        d_pca=Principal_Component_Analysis(l,x)
        reconstructed_data=Reconstruction(d_pca)
        MSE=np.sum(np.sum(np.square(x_t-reconstructed_data),axis=0))/x_t.shape[1]
        mse.append(MSE)
        k=k+20
    #appending the MSE for d=784
    d.append(784)
    d_pca=Principal_Component_Analysis(784,x)
    reconstructed_data=Reconstruction(d_pca)
    MSE=np.sum(np.sum(np.square(x_t-reconstructed_data),axis=0))/x_t.shape[1]
    mse.append(MSE)
    plt.plot(d,mse)
    plt.ylabel('MSE')
    plt.xlabel('The d values')
    plt.show()
#function to reconstruct the images of digit 8 for the given values of d
def Reconstruct_8(x):
    l=[1, 10, 50, 250, 784]
    for k in l:
        d_pca=Principal_Component_Analysis(k,x)
        reconstructed_data=Reconstruction(d_pca)
        plt.figure()
        plt.imshow(reconstructed_data[:,3480].real.reshape(28,28),cmap='BuPu')
        plt.show()
#function to plot eigenvalues vs d
def Eigenvalue_plot(x):
    val,vec=np.linalg.eig(np.cov(x.T))
    indexes=np.argsort(np.abs(val))[::-1]
    val_sorted=val[indexes]
    l=[]
    for i in range(1,785):
        l.append(i)
    plt.plot(l,val_sorted)
    plt.ylabel('Eigen Values')
    plt.xlabel('The d values')
    plt.show()
#main function to call the above defined functions
def main():
    x, y = loadlocal_mnist(
        images_path='/Users/mohita/Downloads/train-images-idx3-ubyte', 
        labels_path='/Users/mohita/Downloads/train-labels-idx1-ubyte')
    x=x/255
    x_t=x.T
    #printing the shape of the original data.
    print('\n The dimensions of the data before applying PCA is',x.shape)
    transformed_data=Principal_Component_Analysis(50,x)
    #Testing the PCA function for some 50 dimesions and finding the shape of the transformed data.
    print('\n The dimensions of the data after applying PCA for 50 dimensions is',transformed_data.shape)
    k=Proportion_of_Variance(x)
    print('\n The value of d using POV = 95% is ',k)
    Mean_Squared_Error(x)
    Reconstruct_8(x)
    Eigenvalue_plot(x)
   


if __name__== "__main__":
  main()

