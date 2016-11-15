from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import argparse
#import g2NN as gn
import cv2
import zernikemoments as zmm
#import matching_zernike
from PIL import Image
#import zernike as zn
#import filtering_blocks as fb
from PIL import Image, ImageDraw
#import flann as fn

#Constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())


#Load the image and apply SLIC
image = cv2.imread(args["image"])
path = args["image"]
segments = slic(img_as_float(image), n_segments = 10, sigma = 5)
#print type(segments)



#Counting the number of superpixels
number_of_segments = 0
for (i, segVal) in enumerate(np.unique(segments)):
	number_of_segments += 1




#Counting number of rows & columns in segments 2D list
rows = segments.shape[0]
columns = segments.shape[1]
#print "rows=%d columns=%d" % (rows,columns)




#Applying SIFT
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
kps_total = len(kps)

descsTemp = descs

#Num_pixles stores the number of pixels in each superpixel
num_pixels = [0] * number_of_segments
for x in range(0,rows) :
	for y in range(0,columns) :
		temp = segments[x,y]
		num_pixels[temp] += 1

im = Image.open(path); 
draw = ImageDraw.Draw(im)




#Num_keypoints counts the number of keypoints in each superpixel
num_keypoints = [0] * number_of_segments
for i in range(0,kps_total) :
	(p,q) = kps[i].pt
	p = int(round(p))
	q = int(round(q))
	t = segments[q,p]
	num_keypoints[t] += 1
	#draw.line((p,q,p+1,q+1), fill="yellow", width=5)

#im.show()
	
#Showing keypoints in Image (gray)
dummy = np.zeros((1,1))
image=cv2.drawKeypoints(gray,kps,dummy,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',image)



#smooth and non-smooth regions are decided
ratio = [0] * number_of_segments
T = 0.0001
is_smooth = [0] * number_of_segments
for (i, segVal) in enumerate(np.unique(segments)):
	print "segment %d (pixel count = %d, number of keypoints = %d)" % (i,num_pixels[i], num_keypoints[i])
	#total += num_keypoints[i]
	ratio[i] = float(num_keypoints[i])/float(num_pixels[i])
	if(ratio[i] < T) :
		is_smooth[i] = 1




#Finding all the indices in the keypoint regions
non_smooth_kps = []
kp_vs_segmentNum = []
num_non_smooth_kps = 0
for i in range(0,kps_total) :
	(p,q) = kps[i].pt
	p = int(round(p))
	q = int(round(q))
	t = segments[q,p]			 # t stores segment value in which i'th keypoint exists
	kp_vs_segmentNum.append(t)
	if(is_smooth[t] == 0) :
		#print "heyu "
		non_smooth_kps.append(i)
		num_non_smooth_kps += 1 #stores no. of keypoint in non smooth segments

print "Number of non smooth keypoints : %d" % (num_non_smooth_kps)

"""
#To create desc_2 storing only non smooth regions' keypoints' descriptors
descs_2 = np.zeros(shape=(num_non_smooth_kps,128))
for i in range(0,len(non_smooth_kps)) :
	p = non_smooth_kps[i]
	descs_2[i] = descs[p]
"""

#To create d1 and d2 storing non smooth regions' keypoints' descriptors(d1=one superpixel descriptors & d2=rest superpixel descriptors)


total_matched=[]


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
 
flann = cv2.FlannBasedMatcher(index_params,search_params)

"""if(descs.all()==descsTemp.all()):
	print "hello" """
   
#matches = flann.knnMatch(descsTemp,descsTemp,k=2)

#d1 = np.zeros(shape=(num_non_smooth_kps,128))
#d2 = np.zeros(shape=(num_non_smooth_kps,128))

#raw_matches=flann.knnMatch(np.asarray(desc1,np.float32),np.asarray(desc2,np.float32), 2) #


for (i, segVal) in enumerate(np.unique(segments)):
	if(is_smooth[i] != 1):
		d1 = np.zeros(shape=(num_non_smooth_kps,128))
		d2 = np.zeros(shape=(num_non_smooth_kps,128))
		#if(type(d1)!=CV_32F):
		#d1.convertTo(d1, CV_32F);
		

		#if(type(d2)!=CV_32F):
		#d2.convertTo(d2, CV_32F);
			
		#d1 = descs
		#d2 = descs
		k1=[]
		k2=[]
		c1=0
		c2=0
		for j in range(0,len(non_smooth_kps)) :
			p = non_smooth_kps[j]
			pos = kp_vs_segmentNum[p]
			if(pos==i):
				d1[c1]=descs[p]
				c1=c1+1	
				k1.append(kps[p])		
			else:
				d2[c2]=descs[p]
				k2.append(kps[p])
				c2=c2+1
		#print c1
		#print c2
		df1 = np.zeros(shape=(c1,128))
		df2 = np.zeros(shape=(c2,128))

		for k in range(0,c1):
			df1[k]=d1[k]
		for k in range(0,c2):
			df2[k]=d2[k]
		
		#matches = flann.knnMatch(d1,d2,k=2)
		matches=flann.knnMatch(np.asarray(d1,np.float32),np.asarray(d2,np.float32), 2) #
		total_matched.append(matches)
	
		#print matches
		# Need to draw only good matches, so create a mask
		matchesMask = [[0,0] for i in xrange(len(matches))]
		   
		# ratio test as per Lowe's paper
		for i,(m,n) in enumerate(matches):
		     if m.distance < 10:
			 matchesMask[i]=[1,0] 

		draw_params = dict(matchColor = (0,255,0),
				   singlePointColor = (255,0,0),
				   matchesMask = matchesMask,
				   flags = 0)
		   
		img3 = cv2.drawMatchesKnn(image,k1,image,k2,matches,None,**draw_params)
		   
		plt.imshow(img3,),plt.show()

	break


#print "Number of non-smooth regions ke keypoints : %d (from %d)" % (num_non_smooth_kps, len(kps))

#G2NN
#matched = []
#matched = gn.g2NN_matching(descs_2,is_smooth)





#print matched
"""for (i, segVal) in enumerate(np.unique(segments)):
	#print "segment %d (pixel count = %d, number of keypoints = %d) ratio = %f is_smootf = %d" % (i,num_pixels[i], num_keypoints[i], ratio[i], is_smooth[i])
	if(is_smooth[i] == 0):
		#for keypoint matching
		gn.g2NN_matching(descs,is_smooth)

#print "Total number of Keypoints len(kps) = %d compared to calculated num = %d" % (kps_total, total)"""


"""For smooth regions: -----------------"""
"""
#For Smooth regions/ blocked based approach begins
zernike_values = []   # storing all zernike values
indices = []  # to store indices for smooth segments

print "Zernike"
for (i, segVal) in enumerate(np.unique(segments)):
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 255
	app=cv2.bitwise_and(image, image, mask = mask)

	#img = Image.fromarray(app, 'RGB')
	mom_val=[]

	

	if(is_smooth[i]==1):
		mom_val=(zn.cal_zernike(app)).tolist()  # to convert ndarray into list
		zernike_values.append(mom_val)
		indices.append(i)
		print mom_val
		im = Image.fromarray(app)
		im.save("new.png")
		im.show()

print len(zernike_values)"""
