import os
import repl, Canny, filters
import pydicom
import cv2
from pydicom.data import get_testdata_files
from skimage import exposure
import numpy as np
from PIL import Image




def equalize_histogram(img):
    img_equalized = exposure.equalize_hist(img)
    return img_equalized


def enhanced_contrast(img):
    img_enhanced = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_enhanced



#check for all the jpg in the folder

imgs = []

#load into imgs all .tif files in the directory
for file_name in sorted(os.listdir('.')):
    if file_name[-3:]=='tif': #checks for mp4 extension
        imgs.append(file_name)



for i in imgs:
    #delete 'imagen.jpg' from working directory
    try:
        os.remove('imagen.png')
        os.remove('eq.png')
        os.remove('eq1.png')
        os.remove('contrast.png')
        os.remove('Equalized_Histogram.png')
        os.remove('Enhanced_Contrast.png')
    except:
        pass

    # ds = pydicom.dcmread(i)
    # dcm_sample = ds.pixel_array
    # # dcm_sample = exposure.equalize_adapthist(dcm_sample)
    # dcm_sample = (np.maximum(dcm_sample,0)/dcm_sample.max()) * 255.0
    # dcm_sample = np.uint8(dcm_sample)

    #for an image, not a  dicom file
    dcm_sample = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

    img = Image.fromarray(dcm_sample)
    img.save('imagen.png')

    equalized = equalize_histogram(dcm_sample)
    equalized *= 255.0
    equalized = np.uint8(equalized)
    img = Image.fromarray(equalized)
    img.save('Equalized_Histogram.png')

    contrast = enhanced_contrast(dcm_sample)
    contrast *= 255.0
    contrast = np.uint8(contrast)
    img = Image.fromarray(contrast)
    img.save('Enhanced_Contrast.png')

    repl.main('imagen.png')
    repl.main('Equalized_Histogram.png')
    repl.main('Enhanced_Contrast.png')

    Canny.main('imagen.png')
    Canny.main('Equalized_Histogram.png')
    Canny.main('Enhanced_Contrast.png')

    filters.main('imagen.png')
    filters.main('Equalized_Histogram.png')
    filters.main('Enhanced_Contrast.png')









    # repl.main(i)
    # Canny.main(i)
    # wt.main(i)
    