
import numpy as np
import cv2 as cv
import math

img = cv.imread("noise.jpg", 0)

print(img.shape)
height, width= img.shape


struct_element = [[0, 255, 0], [255,255,255], [0,255,0]]
struct_element = np.array(struct_element)
print(struct_element)


img_pad = np.pad(img, pad_width=1, mode='constant', constant_values=0)
img_pad1 = np.pad(img, pad_width=1, mode='constant', constant_values=0)
img_pad2 = np.pad(img, pad_width=1, mode='constant', constant_values=0)

dilation = np.pad(img, pad_width=1, mode='constant', constant_values=0)
erosion = np.pad(img, pad_width=1, mode='constant', constant_values=0)
print(img_pad.shape)

h_pad,w_pad = img_pad.shape
#print(struct_element[1][1])

def doErosion(img_pad):
	erosion = np.zeros(img_pad.shape)
	for i in range(0,h_pad-2):
		for j in range(0,w_pad-2):
			if(img_pad[i+1][j+1]==struct_element[1][1] and img_pad[i][j+1]==struct_element[0][1] 
				and img_pad[i+1][j]==struct_element[1][0] and img_pad[i+1][j+2]==struct_element[1][2] and img_pad[i+2][j+1]==struct_element[2][1]):

				erosion[i+1][j+1]=255
			else:
				erosion[i+1][j+1]=0
				erosion[i][j+1] = 0
				erosion[i+1][j] = 0
				erosion[i+1][j+2] = 0
				erosion[i+2][j+1] = 0
				erosion[i][j] = 0
				erosion[i][j+2] = 0
				erosion[i+2][j] = 0
				erosion[i+2][j+2] = 0
	return erosion			




def doDilation(img_pad):
	h,w=img_pad.shape
	dil = np.zeros(img_pad.shape)

	for i in range(0,h-2):
		for j in range(0,w-2):
			if(img_pad[i+1][j+1]==struct_element[1][1]):
				dil[i][j+1] = 255
				dil[i+1][j] = 255
				dil[i+1][j+1] = 255
				dil[i+1][j+2] = 255
				dil[i+2][j+1] = 255
				#dilation[i][j] = 0
				#dilation[i][j+2] = 0
				#dilation[i+2][j] = 0
				#dilation[i+2][j+2] = 0
	return dil			



def doOpening(img_pad):

	erode = doErosion(img_pad)
	#cv.imwrite("erosion"+'.png',erode)
	opening = doDilation(erode)
	opening = opening[2:,:-2]
	cv.imwrite("res_noise1"+'.jpg',opening)


def doCLosing(img_pad):
	dilate = doDilation(img_pad)
	#cv.imwrite("dilation"+'.png',dilate)
	closing = doErosion(dilate)
	closing = closing[2:,:-2]
	cv.imwrite("res_noise2"+'.jpg',closing)

def doBoundaryExtraction():
	task1 = cv.imread("res_noise1.jpg",0)
	task2 = cv.imread("res_noise2.jpg",0)

	task1_pad = np.pad(task1, pad_width=1, mode='constant', constant_values=0)
	task2_pad = np.pad(task2, pad_width=1, mode='constant', constant_values=0)

	task1_erode = doErosion(task1_pad)
	task2_erode = doErosion(task2_pad)
	task1_bound = task1_pad - task1_erode
	task2_bound = task2_pad - task2_erode
	task1_bound = task1_bound[2:,:-2]
	task2_bound = task2_bound[2:,:-2]

	cv.imwrite("res_bound1"+'.jpg',task1_bound)
	cv.imwrite("res_bound2"+'.jpg',task2_bound)



#erode = doErosion(img_pad,erosion)
#dilate = doDilation(img_pad1,dilation)

#cv.imwrite("dilation"+'.png',dilate)
#cv.imwrite("erosion"+'.png',erode)
doOpening(img_pad)
doCLosing(img_pad)
doBoundaryExtraction()


#cv.imwrite("test"+'.png',task1)

