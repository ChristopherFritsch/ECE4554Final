import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy as sp
from tensorflow import keras
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def loadImages(path):
    imgs = []
    for f in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path,f), 0))
    return imgs

def preprocessing(img):
    norm = img
    return norm

def calcGrad(img):
    """
    Calculates angle and magnitude of image using 1D Sobel operators.
    Parameters:
        img - image to calculate angle and magnitude

    Returns:
        angle - image containing angles from given img
        magnitude - image containing magnitudes from given img
    """
    
    # pad image for sobel operators
    img = np.pad(img,[1,1], mode='constant')
    
    # sobel kernels
    sobel_x = np.array([[1,0,-1]])
    sobel_y = np.array([[1],[0],[-1]])
    
    # initialize gradients in x and y direction, and magnitude/angle images
    grad_x = np.zeros(img.shape)
    grad_y = np.zeros(img.shape)
    magnitude = np.zeros(img.shape)
    angle = np.zeros(img.shape)
    
    height, width = img.shape
    
    for r in range(1,height-1):
        for c in range(1,width-1):
            # compute gradient in x direction and y direction
            grad_x[r,c] = np.multiply(img[r,c-1:c+2],sobel_x).sum()
            grad_y[r,c] = np.multiply(img[r-1:r+2,c],sobel_y).sum()
            
            # compute angle and convert from radians to degrees
            angle[r,c] = np.arctan2(grad_y[r,c],grad_x[r,c])*(180/np.pi)
            
            # fix angle to be positive
            if angle[r,c] < 0:
                angle[r,c] += 180
            
            # compute magnitude
            magnitude[r,c] = np.sqrt(grad_x[r,c]**2 + grad_y[r,c]**2)
    
    # angle and magnitude extracted from padded angle and magnitude
    angle = angle[1:height-1,1:width-1]
    magnitude = magnitude[1:height-1,1:width-1]
        
    # --> uncomment below to show angle and magnitude images <-- #
    # cv2.imshow('Magnitude & Angle', np.concatenate((magnitude, angle),axis=1))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return angle, magnitude

def binInterpolate(ang, mag, hist_bin, bin_max):
    """
    Calculates interpolation for histogram bins.
    Parameters
        ang - angle to checked for interpolation 
        mag - magnitude to add to the respective bin
        hist_bin - bin index for angle given
        bin_max - max value for bin, used to calcualte a midpoint for bin

    Returns
    temp_hist - temporary histogram vector to be added to the full histogram later
    """
    temp_hist = np.zeros(9)
    midpoint = bin_max/2
    
    if ang <= midpoint/2:
        temp_hist[hist_bin] = mag
    elif ang > midpoint/2 and ang < midpoint:
        temp_hist[hist_bin] = 0.75*mag
        if hist_bin == 8:
            temp_hist[0] = 0.25*mag
        else:
            temp_hist[hist_bin+1] = 0.25*mag
    elif ang >= midpoint and ang <= midpoint*(3/2):
        temp_hist[hist_bin] = 0.5*mag
        temp_hist[hist_bin+1] = 0.5*mag
        if hist_bin == 8:
            temp_hist[0] = 0.5*mag
        else:
            temp_hist[hist_bin+1] = 0.5*mag
    else:
        temp_hist[hist_bin] = 0.25*mag
        if hist_bin == 8:
            temp_hist[0] = 0.75*mag
        else:
            temp_hist[hist_bin+1] = 0.75*mag
        
    return temp_hist

def histHelper(cell_ang, cell_mag):
    """
    Parameters:
        cell_ang - (height x width) cell of angles extracted from original image
        cell_mag - (height x width) cell of magnitudes extracted from original image

    Returns
        histogram - histogram from given cell (1x9)
    """
    histogram = np.zeros(9)
    
    for r in range(cell_ang.shape[0]):
        for c in range(cell_ang.shape[1]):
            angle = cell_ang[r,c]
            mag = cell_mag[r,c]
            
            if angle < 20:
                histogram += binInterpolate(angle,mag,0,20)
            elif angle >= 20 and angle < 40:
                histogram += binInterpolate(angle,mag,1,40)
            elif angle >= 40 and angle < 60:
                histogram += binInterpolate(angle,mag,2,60)
            elif angle >= 60 and angle < 80:
                histogram += binInterpolate(angle,mag,3,80)
            elif angle >= 80 and angle < 100:
                histogram += binInterpolate(angle,mag,4,100)
            elif angle >= 100 and angle < 120:
                histogram += binInterpolate(angle,mag,5,120)
            elif angle >= 120 and angle < 140:
                histogram += binInterpolate(angle,mag,6,140)
            elif angle >= 140 and angle < 160:
                histogram += binInterpolate(angle,mag,7,160)
            else:
                histogram += binInterpolate(angle,mag,8,180)
    
    return histogram

def calcHist(angle, magnitude, cell_size):
    """
    Calculates histogram for each cell. Each cell is of size (height,width).
    Each block's histogram contains 9 bins. Angles range from 0-180, each bin will
    consider 20 degrees (0,20,40,...,160). The angle present at a particulal pixel
    location will determine the bin, the magnitude at the same location determines
    what value goes into the bin. The magnitude will be shared between adjacent
    bins if the angle present is not at the center of the bin.
    Parameters:
        angle - array consisting of the angles computed from
                original image's gradients
        magnitude - array consisting of the magnitude computed from
                    original image's gradients
        height - specified height of cell to calculate histogram
        width - specified width of cell to calcualte histogram
    Returns
        histograms - N x 1 vector containing histogram information from each
                     cell. N = (# of cells possible in image) * 9.
    """
    hist_vector = []
    
    for r in range(0, angle.shape[0]-1, cell_size):
        for c in range(0, angle.shape[1]-1, cell_size):
            cell_angle = angle[r:r+cell_size,c:c+cell_size]
            cell_magnitude = magnitude[r:r+cell_size,c:c+cell_size]
            hist_vector.append(histHelper(cell_angle,cell_magnitude))
    
    histograms = np.array(hist_vector).flatten()
    histograms = histograms.reshape((histograms.shape[0],1))
    return histograms

def normHist(hist_vector, cell_size, img_height, img_width):
    """
    Normalizes the histograms from each cell by grouping cells into larger
    spatial blocks and contrast normalizing each block seperately. Each of the
    larger spatial blocks is appended to a larger vector. This normalization
    scheme is used to build the final descriptor for classification.
    Parameters: 
    hist_vector - vector containing histogram of gradients information for 
                  each cell. This vector has 9*cell_size*(number of cells) elements
                  and only 1 column.           
    cell_size - 1D size of each cell computed for histogram
    img_height - height of gradient array (number of rows)
    img_width - width of gradient arrary (number of columns)

    Returns
    descriptor - feature descriptor for a single image
    """
    
    # number of bins in histogram
    bins = 9
    block_size = cell_size*2
    
    # overlapping position count for histogram normalization
    width_positions = int(img_width/cell_size) - 1
    height_positions = int(img_height/cell_size) - 1
    
    # number of histograms in each row
    hist_per_row = width_positions + 1
    
    norm_vects = []
    norm_vect = np.zeros(int(bins*(block_size**2)/(cell_size**2)))
    
    # traversing histogram vector, making sure to overlap local cells in order
    # to perform contrast normalization
    for r in range(height_positions):
        for c in range(width_positions):
            # values where the correct local histograms are located in the histogram vector
            start_c = c*bins + r*hist_per_row*bins
            end_c = (c+2)*bins + r*hist_per_row*bins
            start_r = (r+1)*hist_per_row*bins + c*bins
            end_r = (r+1)*hist_per_row*bins + (c+2)*bins
            
            #concatenate the histograms needed
            norm_vect = np.concatenate((hist_vector[start_c:end_c,0],hist_vector[start_r:end_r,0]),axis=0)
            
            # L2 norm used by Navneet Dalal and Bill Triggs on pg.6 (http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
            eps = np.finfo(float).eps
            norm_vect = norm_vect/(np.sqrt((norm_vect**2).sum()+eps**2))
            norm_vects.append(norm_vect)
    
    descriptor = np.array(norm_vects).flatten()
    descriptor = descriptor.reshape((descriptor.shape[0],1))
    return descriptor

def trainSvm(X_train, y_train):
    """
    Train a support vector machine using the HOG features of an image.

    parameters:
    X_train - a training set of HOG features
    y_train - labels for each HOG feature

    Returns: 
    model - a SVM model for the classification of the images
    """
    grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)
    grid.fit(X_train, y_train)
    grid.best_score_

    model = grid.best_estimator_
    model.fit(X_train, y_train)
    return model

def testSvm(img, model):
    patches = sliding_window_view(img, [96, 96])
    labels = []
    locs = []
    for i in range(0, patches.shape[0], 40):
        for j in range(0, patches.shape[1], 40):
            prepro = preprocessing(patches[i,j,:,:])
            angle, magnitude = calcGrad(prepro)
            cell_size = 5
            hist = calcHist(angle, magnitude, cell_size)
            descriptor = np.transpose(normHist(hist, cell_size, prepro.shape[0], prepro.shape[1]))
            
            labels.append(model.predict(descriptor))
            if labels[-1] == 1:
                locs.append((j,i))
    
   
    return labels, locs

def clickEvent(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, '  ', y)

def resizeImage(img):
    return cv2.resize(img, (96, 96))

def testImage(img):
    cv2.imshow('OG Image', img)
    cv2.setMouseCallback('OG Image',clickEvent)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def main():
    # img = cv2.imread("pistol.jpg", 0)
    # testImage(img)
    # crop = img[175:425, 107:357].copy()
    # prepro = preprocessing(crop)
    # angle, magnitude = calcGrad(prepro)
    # height = width = 5
    # hist = calcHist(angle, magnitude, height, width)
    # descriptor = normHist(hist, height, crop.shape[0], crop.shape[1])
    
    knifeImgs = loadImages("./dataset/positive_knife_samples")
    negativeImgs = loadImages("./dataset/negative_samples")

    testImgs = loadImages("./dataset/knife_test_samples")

    imgs = knifeImgs + negativeImgs
    targets = np.concatenate((np.ones(len(knifeImgs)), np.zeros(len(negativeImgs))))
    features = []
    it = 0
    for i in range(len(imgs)):
        print(it)
        resz = resizeImage(imgs[i])
        prepro = preprocessing(resz)
        angle, magnitude = calcGrad(prepro)
        cell_size = 5
        hist = calcHist(angle, magnitude, cell_size)
        descriptor = normHist(hist, cell_size, resz.shape[0], resz.shape[1])
        features.append(descriptor)
        it += 1
        
    features = np.resize(features, [len(features), features[0].size])
    model = trainSvm(features, targets)

    for i in range(len(testImgs)):
        labels, locs = testSvm(testImgs[i], model)
        print(labels)
        for j in range(len(locs)):
            cv2.rectangle(testImgs[i], locs[j], (locs[j][0] + 96, locs[j][1] + 96), (255,0,0))
        cv2.imshow("Image", testImgs[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()