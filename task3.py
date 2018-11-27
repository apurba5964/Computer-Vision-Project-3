import cv2 as cv
import numpy as np


hough_image = cv.imread('original_imgs/hough.jpg',0)

sobel_x = [[-1, 0, 1], [-2,0,2], [-1,0,1]]
sobel_y = [[-1, -2, -1], [0,0,0], [1,2,1]]

def getSobelImage(img,sobel):
    height, width= img.shape
    sobelImage=[[0 for col in range(width)] for row in range(height)]
    for x in range(1,height-1):
        for y in range(1,width-1):
            pixel_x =           (sobel[0][0] * img[x-1][y-1]) + \
                            (sobel[0][1] * img[x-1][y]) + \
                            (sobel[0][2] * img[x-1][y+1]) + \
                            (sobel[1][0] * img[x][y-1])   +\
                             (sobel[1][1] * img[x][y])   + \
                             (sobel[1][2] * img[x][y+1]) + \
                             (sobel[2][0] * img[x+1][y-1]) + \
                             (sobel[2][1] * img[x+1][y]) + \
                             (sobel[2][2] * img[x+1][y+1])
            sobelImage[x][y]=pixel_x
    return np.asarray(sobelImage)        
      
def normalizeMatrix(img):
    h,w=img.shape
    currMax=0
    for x in range(0,h):
        for y in range(0,w):
            if (img[x][y]<0):
                img[x][y]= 0-img[x][y]
            if (currMax<img[x][y]):
                currMax=img[x][y]

    for i in range(0,h):
        for j in range(0,w):
            img[i][j]=(img[i][j]/currMax)*255

    return img


def generateHoughLines(image):
	theta = np.deg2rad(np.arange(-90.0, 90.0))
	print(image.shape)
	width,height = image.shape
	len_diagonal = int(np.ceil(np.sqrt(width * width + height * height)))
	rhos = np.linspace(-len_diagonal, len_diagonal, len_diagonal * 2.0)

	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	len_thetas = len(theta)

	print(len_diagonal)
	accumulator = np.zeros((2 * len_diagonal, len_thetas), dtype=np.uint64)
	y_idx, x_idx = np.nonzero(image)
	print(len_thetas)

	for i in range(len(x_idx)):
		x = x_idx[i]
		y = y_idx[i]

		for t_idx in range(len_thetas):
			rho = int(round(x * cos_theta[t_idx] + y * sin_theta[t_idx]) + len_diagonal)
			#print(rho)
			accumulator[rho, t_idx] += 1

	return accumulator, theta, rhos		








sobelHoughX=getSobelImage(hough_image,sobel_x)
sobelHoughY=getSobelImage(hough_image,sobel_y)
hough_edge_x=normalizeMatrix(sobelHoughX)
hough_edge_y=normalizeMatrix(sobelHoughY)
accumulator, thetas, rhos = generateHoughLines(hough_edge_x)



'''
rho = rhos[idx / accumulator.shape[1]]
theta = thetas[idx % accumulator.shape[1]]

print(rho, np.rad2deg(theta))
'''

#cv.imwrite("accumulator"+'.jpg',accumulator)
#cv.imwrite("hough_edge_y"+'.jpg',hough_edge_y) 


