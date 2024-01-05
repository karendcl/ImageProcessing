import cv2
import pywt
import matplotlib.pyplot as plt

counter = 1

def set_image_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_haar_n_times(image, n=5):
    _, (cH, cV, _) = pywt.dwt2(image, 'haar') 
    for i in range(n-1):
        reconstructed_image = pywt.waverec2((None, (cH, cV, None)), 'haar')
        _, (cH, cV, _) = pywt.dwt2(reconstructed_image, 'haar')
    return pywt.waverec2((None, (cH, cV, None)), 'haar')   
    
def apply_sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobelx, sobely)

def apply_umbral(image):
    _, image = cv2.threshold(image.astype('uint8'), 0, 255, cv2.THRESH_BINARY)


def main(imgsrc):
    img = cv2.imread(imgsrc)

    haar_r = apply_haar_n_times(img)
    edges = apply_sobel(haar_r)
    apply_umbral(edges)

    # plot two images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title(f'{imgsrc[:-4]}')
    ax2.imshow(edges)
    ax2.set_title('Edges Found')

    ax2.xaxis.set_ticklabels([])
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.yaxis.set_ticklabels([])
    ax2.tick_params(axis='y', which='both', length=0)

    ax1.xaxis.set_ticklabels([])
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.yaxis.set_ticklabels([])
    ax1.tick_params(axis='y', which='both', length=0)

    fig.tight_layout()
    global counter
    fig.savefig(f'Third_Meth_{imgsrc[:-4]}_{counter}.jpg')
    counter += 1
    plt.close(fig)