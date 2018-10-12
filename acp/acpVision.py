import cv2
import numpy as np

class acpVision():
 #Processes a raw Atari images.
 def __init__(self):
     self.widthOffset = 0
     self.heightOffset = 34
     self.targetHeight = 160
     self.targetWidth = 160
     self.targetSize = 84

 def rgb2gray(self, image):
     # Converting rgb2gray
     return np.dot(image[...,:3], [0.299, 0.587, 0.114])

 def resize(self, image):
     # take an image gray with dim 2
     # resize it to 84x84
     assert image.ndim == 2, 'resize only accepts array with dim 2 (use rgb2gray before)'

     crop_to_bounding_boxes = image[self.heightOffset:self.heightOffset+self.targetHeight,\
                    self.widthOffset:self.widthOffset+self.targetWidth]

     output = cv2.resize(crop_to_bounding_boxes,\
             (self.targetSize, self.targetSize), interpolation = cv2.INTER_NEAREST)
     return output

 def process(self, state):
     """
     Args:
         state: A [210, 160, 3] Atari RGB State
     Returns:
         A processed [84, 84] state representing grayscale values.
     """
     grayScale = self.rgb2gray(state)
     return self.resize(grayScale)
