import pywt
import numpy as np
import cv2

def find_edges_wavelet_maxima(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar la transformada wavelet con la wavelet Haar
    coeffs = pywt.dwt2(gray, 'haar')

    # Obtener los coeficientes aproximados y los detalles de la imagen
    cA, (cH, cV, cD) = coeffs

    # Reconstruir la imagen a partir de los coeficientes
    reconstructed_image = pywt.waverec2(coeffs, 'haar')

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

# Cargar la imagen
image = cv2.imread('images.jpg')

# Cargar la imagen en escala de grises
image_sobel = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

edges_sobel = wtmm_sobel(image_sobel)

# Guardar los bordes encontrados
cv2.imwrite('wt_sobel_bordes.jpg', edges_sobel)

# Proyectar los bordes en la imagen original
projected_image_sobel = project_edges(image, edges_sobel)

# Guardar la imagen proyectada
cv2.imwrite('wt_sobel.jpg', projected_image_sobel)


edges_wt = find_edges_wavelet_maxima(image)

# Guardar los bordes encontrados
cv2.imwrite('wt_bordes.jpg', edges_wt)

# Proyectar los bordes en la imagen original
projected_image_wt = project_edges(image, edges_wt)

# Guardar la imagen proyectada
cv2.imwrite('wt.jpg', projected_image_wt)

print("Imagenes guardada exitosamente.")
