#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:26:59 2021

@author: rysul
"""
import numpy as np
import math
import cv2 as cv


def showImage(imgMatrix, name):
    cv.imshow(name, imgMatrix)
    cv.waitKey(0)
    cv.destroyAllWindows()


def add_gaussian_noise(mean,variance,image):
      row,col= image.shape
      st_dev = math.sqrt(variance)
      gauss_noise = np.random.normal(mean,st_dev,(row,col))
      gauss_noise = gauss_noise.reshape(row,col)
      noisy = image + gauss_noise
      noisy = noisy.clip(0,255).astype(np.uint8)
      return noisy    

def add_saltpepper_noise(image,p=0.10, s_vs_p=0.50):
    row,col = image.shape
    image_plus_noise = np.copy(image)
    
    # Salt mode
    num_salt = np.ceil(p * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image_plus_noise[coords] = 255
    
    # Pepper mode
    num_pepper = np.ceil(p* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image_plus_noise[coords] = 0
    image_plus_noise = image_plus_noise.clip(0,255).astype(np.uint8)
    return image_plus_noise


def avgFiltering(dim, img):
    # create unweighted masks
    mask = np.ones((dim, dim), dtype = int)
    maskSize = dim * dim
    
    # convolution
    m = math.floor(dim/2)
    n = math.floor(dim/2)
           
    
    convolutedSlicing = img.copy()
    
    for i in range(m, img.shape[0]-m):
        for j in range(n,img.shape[1]-m):
            placeHolder = np.sum(np.multiply(img[i-m:i+m+1, j-n:j+n+1], mask[:2*m+1, :2*n+1]))/maskSize
            convolutedSlicing[i, j] = placeHolder.clip(0, 255).astype(np.uint8)
   
    # from opencv functions

    # using unweighted mask
    convcv = cv.filter2D(img, -1, mask/maskSize)
           
    return convolutedSlicing, convcv

    
def weightedAvgFiltering(dim, img):
    # create weighted mask
    maskW = np.ones((dim, dim), dtype = int)
    maskW[math.floor(dim/2), math.floor(dim/2)] = 2
    maskSizeW = dim * dim + 1
    
    # convolution
    m = math.floor(dim/2)
    n = math.floor(dim/2)
    
    convolutedSlicingW = img.copy()

    for i in range(m, img.shape[0]-m):
        for j in range(n, img.shape[1]-m):
            placeHolder = np.sum(np.multiply(img[i-m:i+m+1, j-n:j+n+1], maskW[:2*m+1, :2*n+1]))/maskSizeW
            convolutedSlicingW[i, j] = placeHolder.clip(0, 255).astype(np.uint8)
    
    # from opencv functions        
    # using weighted mask
    convcvW = cv.filter2D(noisyImageG, -1, maskW/maskSizeW)
            
    return convolutedSlicingW, convcvW


def gaussFiltering(dim, sigma, img):
    # Gaussian mask
    center = math.floor(dim/2)+1
    maskGW = np.zeros((dim, dim), dtype = float)
    for i in range(dim):
        for j in range(dim):
            maskGW[i, j] = math.exp(-((i+1-center)**2 + (j+1-center)**2)/(2 * (sigma**2)))
    maskGW = maskGW/np.sum(maskGW)
    
    # convolution
    m = math.floor(dim/2)
    n = math.floor(dim/2)
    convolutedSlicingGW = img.copy()

    for i in range(m, img.shape[0]-m):
        for j in range(n, img.shape[1]-m):
            placeHolder = np.sum(np.multiply(img[i-m:i+m+1, j-n:j+n+1], maskGW[:2*m+1, :2*n+1]))
            convolutedSlicingGW[i, j] = placeHolder.clip(0, 255).astype(np.uint8)
       
    # from opencv functions 
    # Gaussian filtering in openCV
    convcvGW = cv.GaussianBlur(img, (dim, dim), sigmaX = sigma, sigmaY = sigma)
            
    return convolutedSlicingGW, convcvGW


def medianFiltering(dim, img):
    # convolution
    m = math.floor(dim/2)
    n = math.floor(dim/2)
    convolutedSlicingMedian = img.copy()
    for i in range(m, img.shape[0]-m):
        for j in range(n, img.shape[1]-m):
            placeHolder = img[i-m:i+m+1, j-n:j+n+1]
            # sort placeholder and get the median or use the numpy median operations
            median = np.median(placeHolder).astype(np.uint8) # no need for any flatten as it is the default one 
            convolutedSlicingMedian[i, j] = median
            
    # Median filtering in openCV
    convcvMedian = cv.medianBlur(img, dim)
        
    return convolutedSlicingMedian, convcvMedian


def sobel(dim, img):
    
    weight = np.array([1, 2, 1]).reshape(3,1)
    deriv = np.array([-1, 0, 1]).reshape(3,1)
    # creating the masks
    maskSobleGx = np.dot(deriv,weight.T).astype(int)
    maskSobleGy = np.dot(weight, deriv.T).astype(int)
    

    edgeExtractor = data.copy()
    
    m = math.floor(dim/2)
    n = math.floor(dim/2)
    
    for i in range(m, img.shape[0]-m):
        for j in range(n, img.shape[1]-m):
            placeHolderx = np.sum(np.multiply(img[i-m:i+m+1, j-n:j+n+1], maskSobleGx[:2*m+1, :2*n+1]))
            placeHoldery = np.sum(np.multiply(img[i-m:i+m+1, j-n:j+n+1], maskSobleGy[:2*m+1, :2*n+1]))
            #placeHolderx = np.square(placeHolderx)
            #placeHoldery = np.square(placeHoldery)
            #placeHolder = np.sqrt(placeHolderx+placeHoldery)
            placeHolder = abs(placeHolderx) + abs(placeHoldery)
            #print(placeHolder)
            edgeExtractor[i, j] = placeHolder.clip(0, 255).astype(np.uint8)
    
    # sobel function from CV
    sobel_64x = cv.Sobel(data,cv.CV_32F,1,0,ksize=dim)
    sobel_64y = cv.Sobel(data,cv.CV_32F,0,1,ksize=dim)
    #abs_64 = np.absolute(sobel_64x) + np.absolute(sobel_64y)
    abs_64, _ = cv.cartToPolar(sobel_64x, sobel_64y)
    edgeExtractorCV = np.uint8(abs_64)
    
    return edgeExtractor, edgeExtractorCV



if __name__ == '__main__':
    # read image     
    img = cv.imread('dataset/1.jpg')
    # make image gray scale
    data= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('outputs/grayImage.jpg', data)

    # add gausian noise to the image
    noisyImageG = add_gaussian_noise(50, 100, data)
    # saving the image
    cv.imwrite('outputs/noisyImageG.jpg', noisyImageG)

    # convolution over an unweighted mask
    filteredImage, filteredImageCV = avgFiltering(5, noisyImageG)
    # saving the image
    cv.imwrite('outputs/avgFiltered.jpg', filteredImage)
    cv.imwrite('outputs/avgFilteredCV.jpg', filteredImageCV)
    
    # convolution over a weighted mask
    weightedfilteredImage, weightedfilteredImageCV = weightedAvgFiltering(5, noisyImageG)
    # saving the image
    cv.imwrite('outputs/weightedavgFiltered.jpg', weightedfilteredImage)
    cv.imwrite('outputs/weightedavgFilteredCV.jpg', weightedfilteredImageCV)
    
    # convolution over a gaussian mask
    gaussianFilteredImage, gaussianFilteredImageCV = gaussFiltering(5, 1, noisyImageG)
    # saving the image
    cv.imwrite('outputs/gaussianavgFiltered.jpg', gaussianFilteredImage)
    cv.imwrite('outputs/gaussianavgFilteredCV.jpg', gaussianFilteredImageCV)
    
    # add salt and pepper noise to the image
    noisyImageSP = add_saltpepper_noise(data)
    # saving the image
    cv.imwrite('outputs/noisyImageSP.jpg', noisyImageSP)
    
    # convolution over a median mask with dimension 3x3
    medianFilteredImage3, medianFilteredImage3CV = medianFiltering(3, noisyImageSP)
    # saving the image
    cv.imwrite('outputs/medianFiltered3.jpg',  medianFilteredImage3)
    cv.imwrite('outputs/medianFiltered3CV.jpg',  medianFilteredImage3CV)
    
    # convolution over a median mask with dimension 5x5
    medianFilteredImage5, medianFilteredImage5CV = medianFiltering(5, noisyImageSP)
    # saving the image
    cv.imwrite('outputs/medianFiltered5.jpg', medianFilteredImage5)
    cv.imwrite('outputs/medianFiltered5CV.jpg', medianFilteredImage5CV)
    

    # sobel edge extractor with a 3x3 mask
    edgeImage, edgeImageCV = sobel(3, data)
    cv.imwrite('outputs/edgeExtracted.jpg', edgeImage)
    cv.imwrite('outputs/edgeExtractedCV.jpg', edgeImageCV)

        
