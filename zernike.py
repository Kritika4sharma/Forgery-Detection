import zernikemoments as zmm
import numpy as np
import argparse
import cPickle
import glob
import cv2
from PIL import Image

def cal_zernike(image):
	#image = cv2.imread(img)
	desc = zmm.ZernikeMoments(21)
	index = {}

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	image = cv2.copyMakeBorder(image, 15, 15, 15, 15,
		cv2.BORDER_CONSTANT, value = 255)

	thresh = cv2.bitwise_not(image)
	thresh[thresh > 0] = 255

	"""outline = np.zeros(image.shape, dtype = "uint8")
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	cv2.drawContours(outline, [cnts], -1, 255, -1) """

	moments = desc.describe(image)
	
	moments2 = desc.describe(outline)
	#index[pokemon] = moments
	"""f = open(args["index"], "w")
	f.write(cPickle.dumps(index))
	f.close()"""
	#print moments
	
	return moments2


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, help = "Path to the image")
	args = vars(ap.parse_args())

	image = cv2.imread(args["image"])
	val=cal_zernike(image)
	print val


