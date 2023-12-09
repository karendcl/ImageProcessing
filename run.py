import os
import repl, Canny, wt

import matplotlib.pyplot as plt
import pydicom
import cv2
from pydicom.data import get_testdata_files
from skimage import exposure
import numpy as np
from PIL import Image



#check for all the jpg in the folder

imgs = []

#load into videos all .mp4 files in the directory
for file_name in sorted(os.listdir('.')):
    if file_name[-3:]=='dcm': #checks for mp4 extension
        imgs.append(file_name)



for i in imgs:
    #delete 'imagen.jpg' from working directory
    try:
        os.remove('imagen.png')
    except:
        pass

    ds = pydicom.dcmread(i)
    dcm_sample = ds.pixel_array
    # dcm_sample = exposure.equalize_adapthist(dcm_sample)
    dcm_sample = (np.maximum(dcm_sample,0)/dcm_sample.max()) * 255.0
    dcm_sample = np.uint8(dcm_sample)

    img = Image.fromarray(dcm_sample)
    img.save('imagen.png')

   

    # cv2.imshow(' ', dcm_sample)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    # cv2.imwrite('imagen.png', dcm_sample)
    repl.main('imagen.png')
    Canny.main('imagen.png')
    wt.main('imagen.png')








    # repl.main(i)
    # Canny.main(i)
    # wt.main(i)
    