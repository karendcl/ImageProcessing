import cv2 


def CannyEdge(imgpath):
    img = imgpath


    # Setting parameter values 
    t_lower = 80  # Lower Threshold 
    t_upper = 150  # Upper threshold 
  
# Applying the Canny Edge filter 
    edge = cv2.Canny(img, t_lower, t_upper)
    return edge
  
