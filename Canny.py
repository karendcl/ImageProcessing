import cv2 
import pywt
import numpy as np
import matplotlib.pyplot as plt

def ProjectEdges(img_orig, edges):
    img = img_orig.copy()
    img[edges > 10] = [0, 255, 0]  # Asignar verde (BGR) a los bordes

    return img

def CannyEdge(img):
    # Setting parameter values 
    t_lower = 40  # Lower Threshold 
    t_upper = 180  # Upper threshold 
  
# Applying the Canny Edge filter and maintain same size
    edge = cv2.Canny(np.uint8(img), t_lower, t_upper)
   
    return edge
  
def main(img):
    img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(gray,'haar')

    cA, (cH, cV, cD) = coeffs

    edge = CannyEdge(cA)

    edge = cv2.resize(edge, (cH.shape[1], cV.shape[0]))

    edge = np.clip(edge,0,255)



    reconstructed = pywt.idwt2((edge,(cH,cV,cD)),'haar')
    reconstructed = np.clip(reconstructed,0,300)
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

    ax1.xaxis.set_ticklabels([])
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.yaxis.set_ticklabels([])
    ax1.tick_params(axis='y', which='both', length=0)

    ax2.xaxis.set_ticklabels([])
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.yaxis.set_ticklabels([])
    ax2.tick_params(axis='y', which='both', length=0)

    ax3.xaxis.set_ticklabels([])
    ax3.tick_params(axis='x', which='both', length=0)
    ax3.yaxis.set_ticklabels([])
    ax3.tick_params(axis='y', which='both', length=0)

    fig.tight_layout()
    plt.show()
