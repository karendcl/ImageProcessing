import cv2 
import pywt
import numpy as np
import matplotlib.pyplot as plt


counter = 1
def CannyEdge(img):
    # Setting parameter values 
    t_lower = 150  # Lower Threshold
    t_upper = 250  # Upper threshold
  
# Applying the Canny Edge filter and maintain same size
    edge = cv2.Canny(np.uint8(img), t_lower, t_upper)
   
    return edge
  
def main(imgsrc):
    global counter
    img = cv2.imread(imgsrc)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(gray,'db38')

    cA, (cH, cV, cD) = coeffs

    edge = CannyEdge(cA)

    edge = cv2.resize(edge, (cH.shape[1], cV.shape[0]))

    edge = np.clip(edge,0,255)



    reconstructed = pywt.idwt2((edge,(cH,cV,cD)),'db38')
    reconstructed = np.clip(reconstructed,0,255)
    reconstructed = reconstructed.astype(np.uint8)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2BGRA)


    #plot two images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title(f'{imgsrc[:-4]}')
    ax2.imshow(reconstructed)
    ax2.set_title('Edges Found')


    ax1.xaxis.set_ticklabels([])
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.yaxis.set_ticklabels([])
    ax1.tick_params(axis='y', which='both', length=0)

    ax2.xaxis.set_ticklabels([])
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.yaxis.set_ticklabels([])
    ax2.tick_params(axis='y', which='both', length=0)

    fig.tight_layout()
    global counter
    fig.savefig(f'Canny_{imgsrc[:-4]}_{counter}.jpg')
    counter += 1
    plt.close(fig)
