# Image Processing

This project contains the implementation of the three methods to approximate edges mentioned in "EDGE DETECTION IN MEDICAL IMAGES
USING THE WAVELET TRANSFORM" by J. Petrová and E. Hostálková, applied to mammography images.

+ Replacing Approximation Coefficients from the wavelet decomposition by 0 and reconstructing the image from the remaining coefficients

+ Applying the Canny edge detection algorithm to the Approximation Coefficients from the wavelet decomposition and reconstructing the image using the edge image as the new Approximation Coefficients.

+ The Wavelet Transform Modulus Maxima Method
> This method was developed by Stephane Mallat. Its principle is based on finding local
maxima of wavelet coefficients, which represent the edges in the image. The method uses only horizontal and vertical coefficients values
(LowHigh and HighLow coefficients 1) from each level of wavelet decomposition.
