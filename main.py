import numpy as np
import keras as ks
import scipy as sp
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def loadImages(path):
    imgs = []
    for f in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path,f), cv2.COLOR_RGB2GRAY))
    return imgs

def preprocessing(img):
    norm = cv2.normalize(img)
    return norm

def calcGrad(img):
    return img

def calcHist(img):
    return img

def calcHogVec(img):
    return img

def trainSvm(X_train, y_train):
    grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)
    grid.fit(X_train, y_train)
    grid.best_score_

    model = grid.best_estimator_
    model.fit(X_train, y_train)
    return model

def testSvm(img, model):
    prepo = preprocessing(img)
    grad = calcGrad(prepo)
    hist = calcHist(grad)
    hog = calcHogVec(hist)
    labels = model.predict(hog)
    return labels

def main():
    imgs = loadImages("./dataset")
    features = []
    targets = []
    for img in imgs:
        prepro = preprocessing(img)
        grad = calcGrad(prepro)
        hist = calcHist(grad)
        features.append(calcHogVec(hist))

    model = testSvm(features, targets)
    

if __name__ == "__main__":
    main()