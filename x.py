#return wavelet transform of an image
import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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
        #canny algorithm in the wavelet coefficients
        #create Matlike using LL
    
        gray = cv2.cvtColor(LL, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5) 
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
        #Compute the gradient magnitude and direction for each pixel in the image.
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        graddir = np.arctan2(sobely, sobelx)
        #Apply a non-maximum suppression step to the gradient magnitude image
        mag, ang = cv2.cartToPolar(gradmag, graddir, angleInDegrees=True)
        #Threshold the gradient magnitude image to produce a binary edge map.
        thresh = 30
        binary = np.uint8((ang < thresh))
        #Perform hysteresis thresholding on the binary edge map.
        binary = cv2.bitwise_not(binary)
   

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
    
    img1 = MethodReplacingBy0(img)
    img1.save("MethodReplacingBy0.png")

    img2 = ProjectEdges(img, img1)
    #save img2
    img2.save("Blended.png")


if __name__ =="__main__":
    main()