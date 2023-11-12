#return wavelet transform of an image
import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import Canny

def Method2_1_1(img):
    return MethodReplacingBy0(img)

def Method2_1_2(img):
    img = np.array(img)
    LL, LH,HL,HH = wavelet_transform(img)
    edge = EdgeDetectionUsingCanny(LL)
    cv2.imwrite('CannyEdgeDetection.png', edge)
    img2 = inverse_wavelet_transform(edge,LH,HL,HH)
    img2 = Image.fromarray(np.uint8(img2))
    img2 = img2.convert("L")

    img2.save("CannyBlended.jpg")

def wavelet_transform(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH

def inverse_wavelet_transform(LL, LH, HL, HH):
    coeffs = LL, (LH, HL, HH)
    img = pywt.idwt2(coeffs, 'haar')
    return img

#image preprocessing
def ImageTresholding(img):
    img = np.array(img)
    img[img<128] = 0
    img[img>=128] = 255
    return Image.fromarray(img)

def MedianFiltering(imag):
    imag = np.array(imag)
    for i in range(1, imag.shape[0]-1):
        for j in range(1, imag.shape[1]-1):
            imag[i][j] = np.median(imag[i-1:i+2, j-1:j+2])
    return Image.fromarray(imag)

def Preprocess(imag):
    imag = ImageTresholding(imag)
    imag = MedianFiltering(imag)
    return imag

def ProjectEdges(img:Image, edges:Image):
    #project edges into img
    return Image.blend(img, edges, 0.3)


def EdgeDetectionUsingCanny(LL):
        LL = np.array(np.uint8(LL))
        edge = Canny.CannyEdge(LL)
        return edge
       



   

#Approximation coefficients replaced by 0
def MethodReplacingBy0(img):
    img = np.array(img)
    LL, LH, HL, HH = wavelet_transform(img)
    LL = np.zeros(LL.shape)
    img = inverse_wavelet_transform(LL, LH, HL, HH)
    #convert from array to image
    img = Image.fromarray(np.uint8(img))
    #convert image to black and white
    img = img.convert("L")
    return img


   

def main():
    img = Image.open("img.png")
    img = img.convert("L")

    result1 = Method2_1_1(img)
    result1.save("MethodReplacingBy0.png")

    # img2 = ProjectEdges(img, img1)
    # #save img2
    # img2.save("Blended.png")

    Method2_1_2(img)


    


if __name__ =="__main__":
    main()