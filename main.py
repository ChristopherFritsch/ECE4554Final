import numpy as np
import keras as ks
import scipy as sp
import cv2
import os

def loadImages(path):
    imgs = []
    for f in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path,f), cv2.COLOR_RGB2GRAY))
    return imgs

def preprocessing(img):
    return img

def calcGrad(img):
    return img

def calcHist(img):
    return img

def calcHogVec(img):
    return img

def svm(img):
    return img

def main():
    imgs = loadImages("./dataset")
    for img in imgs:
        prepro = preprocessing(img)
        grad = calcGrad(prepro)
        hist = calcHist(grad)
        hog = calcHogVec(hist)
        out = svm(hog)
    

if __name__ == "__main__":
    main()