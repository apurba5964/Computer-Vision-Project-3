import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("point.jpg", 0)
img_seg = cv.imread("segment.jpg", 0)

#print(img.shape)

height,width = img.shape

mask = [[-1, -1, -1], [-1,8,-1], [-1,-1,-1]]
mask = np.array(mask)

def count_frequency(seq) -> dict:
	
	hist = {}
	for i in seq:
		hist[i] = hist.get(i, 0) + 1
	return hist		


def generateHistogram(image):
	plt.clf()
	img_freq = count_frequency(image.ravel())
	plt.bar(list(img_freq.keys()), img_freq.values(), color='g')
	plt.savefig('histogram_seg.jpg')


def imageMasking(img,mask):
	img_pad = np.pad(img, pad_width=1, mode='constant', constant_values=0)
	img1 = np.zeros(img_pad.shape)
	img2 = np.zeros(img_pad.shape)
	print(img1.shape)
	h,w = img_pad.shape
	for i in range(0,h-2):
		for j in range(0,w-2):
			sume = 0
			for k in range(0,3):
				for l in range(0,3):
					sume = sume + img_pad[i+k][j+l] * mask[k][l]


			img1[i+1][j+1]=sume


					
	#img_pad = img_pad[2:,:-2]
	img1 = abs(img1*255)/np.max(img1)
	img1 = img1[2:,:-2]
	img2 = img2[2:,:-2]
	print(np.max(img1))
	#print(img1.ravel().tolist())
	#generateHistogram(img1)
	#img2 = img1
	print(img1.shape)
	h1,w1= img1.shape
	

	for i in range(0,h1):
		for j in range(0,w1):
			if(img1[i][j]>30):
				img2[i][j]=255
				#print(i,j)

	
	cv.imwrite("testmask"+'.jpg',img1)
	cv.imwrite("testpoint"+'.jpg',img2)	


def getOptimalThreshold(image):
	image = image[np.where( image > 186 )] 
	T=190
	for i in range(0,20):
		print(T)

		G1=[]
		G2=[]
		#print(len(image))
		#h1,w1= image.shape
	
		for i in range(0,len(image)):
			#for j in range(0,w1):
			if(image[i]>T):
				G2.append(image[i])
			else:
				G1.append(image[i])
	
		mu1 = np.mean(G1)
		mu2 = np.mean(G2)
		T_NEW = 0.5 * (mu1+mu2)
		T=T_NEW
	return T				






								
def imageSegmentation(image):
	#generateHistogram(image[np.nonzero(image)])
	threshold=getOptimalThreshold(image[np.nonzero(image)])
	
	image_seg = np.ones(image.shape)*255
	h1,w1= image.shape
	for i in range(0,h1):
		for j in range(0,w1):
			if(image[i][j]>threshold):
				image_seg[i][j]=0
				
	
	
	cv.rectangle(image_seg, (100, 100), (200, 200), (255, 255, 0), 2)
	


	cv.imwrite("testseg"+'.jpg',image_seg)	
			
	#plt.hist(image[np.nonzero(image)].ravel(),256,[0,256]) 
	#plt.show()


					 

#print(img_seg.ravel().tolist())

#generateHistogram(img_seg)
#imageMasking(img,mask)

imageSegmentation(img_seg)






