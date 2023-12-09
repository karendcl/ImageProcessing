import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


def ProjectEdges(img_orig, edges):
    img = img_orig.copy()
    # Asignar valores de color a los bordes utilizando la imagen con los bordes resaltados
    img[edges > 10] = [0, 255, 0]  # Asignar verde (BGR) a los bordes

    return img 

def main(imgsource):

    #convert from ndarray to matlike
    img = cv2.imread(imgsource)


    gray = img
    
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(gray,'haar')

    cA, (cH, cV, cD) = coeffs
    ad = cA
    cA = np.zeros_like(cA)

    reconstructed = pywt.idwt2((None,(cH,cV,cD)),'haar')
    reconstructed = np.clip(reconstructed,0,255)
    proj = ProjectEdges(img, reconstructed)
    reconstructed = reconstructed.astype(np.uint8)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2BGRA)


    #plot two images side by side
    fig ,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(img)
    ax1.set_title('Original')
    ax2.imshow(reconstructed)
    ax2.set_title('Edges Found')
    ax3.imshow(proj)
    ax3.set_title('Edges Projected')

    ax3.xaxis.set_ticklabels([])
    ax3.tick_params(axis='x', which='both', length=0)
    ax3.yaxis.set_ticklabels([])
    ax3.tick_params(axis='y', which='both', length=0)

    ax2.xaxis.set_ticklabels([])
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.yaxis.set_ticklabels([])
    ax2.tick_params(axis='y', which='both', length=0)

    ax1.xaxis.set_ticklabels([])
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.yaxis.set_ticklabels([])
    ax1.tick_params(axis='y', which='both', length=0)


    fig.tight_layout()
    plt.show()




