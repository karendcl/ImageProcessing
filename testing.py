import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_edges_wavelet_maxima(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar la transformada wavelet con la wavelet Haar
    coeffs = pywt.dwt2(gray, 'haar')

    # Obtener los coeficientes aproximados y los detalles de la imagen
    cA, (cH, cV, cD) = coeffs

    # Reconstruir la imagen a partir de los coeficientes
    reconstructed_image = pywt.waverec2((None, (cH, cV, None)), 'haar')

    # Calcular el máximo local en la imagen reconstruida
    maxima = reconstructed_image.max()

    # Aplicar un umbral para obtener los bordes
    _, edges = cv2.threshold(reconstructed_image.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return edges

def wtmm_sobel(image):
    # Aplicar el operador de gradiente de Sobel
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular la magnitud del gradiente
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Aplicar un umbral para obtener los bordes
    _, edges = cv2.threshold(gradient_magnitude.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges

def project_edges(original_image, edges):
    
    img = original_image.copy()
    # Asignar valores de color a los bordes utilizando la imagen con los bordes resaltados
    img[edges > 250] = [0, 255, 0]  # Asignar verde (BGR) a los bordes

    return img

def main(imgsource):
    # Cargar la imagen
    image = cv2.imread(imgsource)


    edges_wt = find_edges_wavelet_maxima(image)

    # Guardar los bordes encontrados
    # cv2.imwrite('wt_bordes.jpg', edges_wt)

    # Proyectar los bordes en la imagen original
    projected_image_wt = project_edges(image, edges_wt)

    # Guardar la imagen proyectada
    # cv2.imwrite('wt.jpg', projected_image_wt)
    

    # Cargar la imagen en escala de grises
    image_sobel = cv2.imread(imgsource, cv2.IMREAD_GRAYSCALE)

    edges_sobel = wtmm_sobel(edges_wt)

    # Guardar los bordes encontrados
    # cv2.imwrite('wt_sobel_bordes.jpg', edges_sobel)

    # Proyectar los bordes en la imagen original
    projected_image_sobel = project_edges(image, edges_sobel)

    # Guardar la imagen proyectada
    # cv2.imwrite('wt_sobel.jpg', projected_image_sobel)

    fig ,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
    ax1.imshow(image)
    ax1.set_title('Original')
    ax2.imshow(cv2.cvtColor(edges_sobel, cv2.COLOR_BGR2RGB))
    ax2.set_title('sobel edges')
    ax3.imshow(projected_image_sobel)
    ax3.set_title('wt sobel')
    ax4.imshow(cv2.cvtColor(edges_wt, cv2.COLOR_BGR2RGB))
    ax4.set_title('wt edges')
    ax5.imshow(projected_image_wt)
    ax5.set_title('wt')

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

    ax4.xaxis.set_ticklabels([])
    ax4.tick_params(axis='x', which='both', length=0)
    ax4.yaxis.set_ticklabels([])
    ax4.tick_params(axis='y', which='both', length=0)

    ax5.xaxis.set_ticklabels([])
    ax5.tick_params(axis='x', which='both', length=0)
    ax5.yaxis.set_ticklabels([])
    ax5.tick_params(axis='y', which='both', length=0)

    fig.tight_layout()
    plt.show()