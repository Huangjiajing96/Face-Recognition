FACE RECOGNITION
----------------

The Yale Face Database contains 165 grayscale images in GIF format of 15 individuals. There are 11 images per
subject, one per different facial expression or configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink. Your tasks are the following:

1. I have divided image into small blocks and extracted local binary patterns (LBP) from each block. Concatenated all LBP histograms to make a feature vector of an image.

2. Another feature vector is created out of gray levels of integral image.
3. Finally gray levels of image have been used as the last feature vector.
4. After Concatenating all feature vectors. I have Taken four images of each person for testing and the rest as training examples.
5. Using PCA to classify image for one-verses all classification scheme, i have shown results for few images that are selected randomly and reported the accuracy for all testing images using individual feature sets (gray level, integral, and LBP separately) and also for concatenated feature sets.