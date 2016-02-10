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
def LBP ( # The function gives the lbp matrix of all the images stored in the specified path
path): # the folder in which the training images are stored
    bi = array([128,64,32,16,8,4,2,1]) # Binary to decimal conversion matrix
    LBPft = [] # initializing the LBP array
    for file in os.listdir(path): # accessing all the training images
        I = array(Image.open(os.path.join(path,file)).convert('L')) # read input images
        T = zeros(shape = (I.shape[0],I.shape[1])) # Initializing LBP image
        for r in range(1,I.shape[0]-1): # rows
            for c in range(1,I.shape[1]-1): # Columns 
                compare = [] # the list stores values of each cell
                for nr,nc in ([r+1,c+1],[r,c+1],[r+1,c-1],[r,c+1],[r,c-1],[r-1,c+1],[r-1,c],[r+1,c-1]): # neighbor pixels
                    compare.append([0, 1][I[nr,nc] > I[r,c]]) # comparing center pixel of each cell with its neighbors
                T[r,c] = sum(bi*array(compare)) # using the decimal value...
        T= T.ravel() # the LBP vector of each image
        LBPft.append(T) # Contains the LBP vectors of all training images
    return array(LBPft).T

def find_eigenspace(vec):
    mean_vec = vec.mean(axis=1) #Computing the average face image
    A = vec.T - mean_vec # Computing the difference image for each image in the training database
    A = A.T
    cov_gray = dot(A.T,A) #the surrogate of covariance matrix C=A*A'.
    e,EV = linalg.eig(cov_gray) # Computing Eigen Values and Eigen Vector....
    e,EV = e[::-1],EV[::-1] # Arranging in descending order
    (init,ratio,j,sumofe) = (0,0.1,-1,sum(e)) # initialing the parameters to be used in while loop.
    while (ratio<=.98):#implementing sum(Eig(sub(i)))/sum(s) for <= 98%
        j+=1
        init = init + e[j]
        ratio = init/sumofe
    j = 105
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
LBPft,Iin_LBP = LBP('train'),LBP('test')
(LBPproj,LBPeig,LBPmean) = find_eigenspace(LBPft)
#=================== Training ==================================== End ========#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#=================== Testing ==================================== Start =======#
LBP_out = test_projection(Iin_LBP,LBPeig,LBPmean)
LBP_acc,true_train = compare(Iin_LBP,LBP_out,LBPproj)
#=================== Testing ==================================== End =======#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
print"\nAccuracy of LBP feature set = %.2f%%\n"%(LBP_acc*100/60.0)
