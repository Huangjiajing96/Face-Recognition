from PIL import Image
import numpy
from matplotlib import pyplot as plt
from pylab import *
import os
import numpy
from numpy import *
from scipy.spatial import distance
import random
#=================== Function Definitions ==================================== Start ======#
def integral ( # The function gives the integral matrix of all the images stored in the specified path
path): # the folder in which the training images are stored
    intgft = [] # initializing the Integral array
    for file in os.listdir(path): # accessing all the training images
        I = array(Image.open(os.path.join(path,file)).convert('L')) # read input images
        T = zeros(shape = (I.shape[0],I.shape[1])) # Initializing Integral image
        T[0,:],T[:,0] = I[0,:],I[:,0]
        for r in range(1,I.shape[0]): # rows
            for c in range(1,I.shape[1]): # Columns 
                T[r,c] = I[r,c] + T[r-1,c] + T[r,c-1] - T[r-1,c-1]
        T= T.ravel() # the Integral vector of each image
        intgft.append(T) # Contains the Integral vectors of all training images
    return array(intgft).T

def find_eigenspace(vec):
    mean_vec = vec.mean(axis=1) #Computing the average face image
    A = vec.T - mean_vec # Computing the difference image for each image in the training database
    A = A.T
    cov_gray = dot(A.T,A) #the surrogate of covariance matrix C=A*A'.
    e,EV = linalg.eig(cov_gray) # Computing Eigen Values and Eigen Vector....
    e,EV = e[::-1],EV[::-1] # Arranging in descending order
    j = 0
    for i in range(len(e)): # Choosing how many prominent features are to be selected
        if e[i] >= 1:
            j+=1
    V = EV[:,0:j] # using only 'j' prominent features...
    Eig = dot(A,V) # defining the eigen space...
    proj = zeros(shape = (Eig.shape[1],A.shape[1]))
    for m in range(A.shape[1]):
        proj[:,m] = dot(Eig.T,A[:,m]) # projecting the training dataset on eigen space...
    return proj,Eig,mean_vec

def test_projection(vec,Eig,mean):
    test_proj = zeros(shape = (Eig.shape[1],vec.shape[1]))
    for m in range(vec.shape[1]):
        diff = vec[:,m].T - mean # taking the mean of input image...
        test_proj[:,m] = dot(Eig.T,diff.reshape(len(diff),1)).T # projecting the testing dataset on eigen space...
    return test_proj

def compare(test_vec,test_proj,train_proj):
    t = [] # stores which image of the training dataset was matched
    tru = 0 # count the number of true images
    true_train = [] # gives the index of matched images from training dataset and testing training dataset
    for z in range(test_vec.shape[1]):
        dist = []
        for n in range(train_proj.shape[1]):
            dist.append(distance.euclidean(test_proj[:,z],train_proj[:,n])) # taking the minimum distance
        t.append(dist.index(min(dist)))
        l = (z/4) # since each person has 4 testing images--> l gives the subject number...
        if t[z] in range((l*7),((l*7)+7)): # for 7 training images per person
            tru+=1 # counting if macthed with the same person....
            true_train.append((z,t[z]))
    random.shuffle(true_train) # choosing 8 random matches....
    return tru,true_train
#=================== Function Definitions ==================================== End ======#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#=================== Training ==================================== Start ========#
integralft,Iin_integral = integral('train'),integral('test')
(integralproj,integraleig,integralmean) = find_eigenspace(integralft)
#=================== Training ==================================== End ========#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#=================== Testing ==================================== Start =======#
integral_out = test_projection(Iin_integral,integraleig,integralmean)
integral_acc,true_train = compare(Iin_integral,integral_out,integralproj)
#=================== Testing ==================================== End =======#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
print"\nAccuracy of Integral feature set = %.2f%%\n"%(integral_acc*100/60.0)