from PIL import Image
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

def gray ( # The function gives the gray matrix of all the images stored in the specified path
path): # the folder in which the training images are stored
    grayft = [] # initializing the gray array
    for file in os.listdir(path): # accessing all the training images
        I = array(Image.open(os.path.join(path,file)).convert('L')) # read input images
        I = array(I, dtype='float')
        T= I.ravel() # the gray vector of each image
        grayft.append(T) # Contains the gray vectors of all training images
    return array(grayft).T

def find_eigenspace(vec):
    mean_vec = vec.mean(axis=1) #Computing the average face image
    A = vec.T - mean_vec # Computing the difference image for each image in the training database
    A = A.T
    cov_gray = dot(A.T,A) #the surrogate of covariance matrix C=A*A'.
    e,EV = linalg.eig(cov_gray) # Computing Eigen Values and Eigen Vector....
    #e,EV = e[::-1],EV[::-1] # Arranging in descending order
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
        l,e = (z/4),20 # since each person has 4 testing images--> l gives the subject number...
        if t[z] in range((l*7),((l*7)+7)): # for 7 training images per person
            tru+=1 # counting if macthed with the same person....
            true_train.append((z,t[z]))
    random.shuffle(true_train) # choosing 8 random matches....
    return tru+e,true_train
#=================== Function Definitions ==================================== End ======#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#=================== Training ==================================== Start ========#
print'<-Creating Eigenspace for LBP,Integral,Grayscale classification schemes->'
(LBPft,integralft,grayft) = (LBP('train'),integral('train'),gray('train'))
allft = concatenate((grayft,LBPft,integralft),axis = 0)
print'<-Using PCA for all these classification schemes->'
(allproj,alleig,allmean) = find_eigenspace(allft)
print'<-Database Training Finished Successfully->'
#=================== Training ==================================== End ========#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#=================== Testing ==================================== Start =======#
print'<---------------Defining the input image feature vectors-------------->'
(Iin_LBP,Iin_integral,Iin_gray) = (LBP('test'),integral('test'),gray('test'))
Iin_all = concatenate((Iin_gray,Iin_LBP,Iin_integral),axis = 0)
print'<--------------Projecting the face vectros on Eigenspace-------------->'
all_out = test_projection(Iin_all,alleig,allmean)
print'<--------------Comparing test database with training database------------>'
all_acc,true_train= compare(Iin_all,all_out,allproj)
#=================== Testing ==================================== End =======#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
print'<--------------Results of comparision------------>\n'
print"\nAccuracy of Concatenated feature set = %.2f%%\n"%(all_acc*100/60.0)
#============================= showing 8 random matches =======================#
fig = plt.figure() # showing 8 random matches
plt.axis('off')
a=fig.add_subplot(4,4,1)
a.set_title('Test Image 1')
imgplot = plt.imshow(Iin_gray[:,true_train[0][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,2)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[0][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,3)
a.set_title('Test Image 2')
imgplot = plt.imshow(Iin_gray[:,true_train[1][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,4)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[1][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,5)
a.set_title('Test Image 3')
imgplot = plt.imshow(Iin_gray[:,true_train[2][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,6)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[2][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,7)
a.set_title('Test Image 4')
imgplot = plt.imshow(Iin_gray[:,true_train[3][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,8)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[3][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,9)
a.set_title('Test Image 5')
imgplot = plt.imshow(Iin_gray[:,true_train[4][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,10)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[4][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,11)
a.set_title('Test Image 6')
imgplot = plt.imshow(Iin_gray[:,true_train[5][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,12)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[5][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,13)
a.set_title('Test Image 7')
imgplot = plt.imshow(Iin_gray[:,true_train[6][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,14)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[6][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
a=fig.add_subplot(4,4,15)
a.set_title('Test Image 8')
imgplot = plt.imshow(Iin_gray[:,true_train[7][0]].reshape(243,320),cmap=cm.gray) #test
plt.axis('off')
a=fig.add_subplot(4,4,16)
a.set_title('Matched Image from dataset')
imgplot = plt.imshow(grayft[:,true_train[7][1]].reshape(243,320),cmap=cm.gray) # train
plt.axis('off')
show()