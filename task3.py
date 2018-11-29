import cv2 as cv
import numpy as np
from math import exp

hough_image = cv.imread('original_imgs/hough.jpg',0)
v_org_image = cv.imread('original_imgs/hough.jpg')
d_org_image = cv.imread('original_imgs/hough.jpg')
circle_org_image = cv.imread('original_imgs/hough.jpg')

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
    #theta = np.deg2rad([-90,-60,-45,-30,-15,0,15,30,45,60,90])
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

def thresholdSobel(Image):
    img = np.zeros(Image.shape)
    h1,w1 = Image.shape
    for i in range(0,h1):
        for j in range(0,w1):
            if(Image[i][j]>40):
                img[i][j]=255
    return img  

def hough_peaks1(H, num_peaks):
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


def hough_lines_draw(img, indicies, rhos, thetas):
    for i in range(len(indicies)):
        
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


sobelHoughX=getSobelImage(hough_image,sobel_x)
sobelHoughY=getSobelImage(hough_image,sobel_y)
hough_edge_x=normalizeMatrix(sobelHoughX)
hough_edge_y=normalizeMatrix(sobelHoughY)

threshold_sobel_x = thresholdSobel(hough_edge_x)

accumulator, thetas, rhos = generateHoughLines(threshold_sobel_x)



peaks = hough_peaks1(accumulator,200)
v_peaks = peaks[peaks[:,1]>87]
v_peaks = v_peaks[v_peaks[:,1]<110]
#v_peaks = v_peaks[v_peaks[:,0]>910]
v_peaks = v_peaks[:40,:]
d_peaks = peaks[peaks[:,1]<56]
d_peaks = d_peaks[d_peaks[:,0]>714]
d_peaks = d_peaks[:20,:]
#print(d_peaks)

#print(v_peaks)
#theta = thetas[peaks[2][1]]
#theta

hough_lines_draw(v_org_image, v_peaks,rhos, thetas)
hough_lines_draw(d_org_image, d_peaks,rhos, thetas)

cv.imwrite("red_line"+'.jpg',v_org_image)
cv.imwrite("blue_lines"+'.jpg',d_org_image)

cv.imwrite("accumulator"+'.jpg',accumulator)


#-----------------------------------------Hough Circle---------------------------#
def thresholdSobelCircle(Image):
    img = np.zeros(Image.shape)
    h1,w1 = Image.shape
    for i in range(0,h1):
        for j in range(0,w1):
            if(Image[i][j]>55):
                img[i][j]=255
    return img

threshold_sobel_y = thresholdSobelCircle(hough_edge_y)

def generateHoughCircles(image):
    radius = 20
    theta = np.deg2rad(np.arange(0.0, 360.0))
    #theta = np.deg2rad([-90,-60,-45,-30,-15,0,15,30,45,60,90])
    print(image.shape)
    width,height = image.shape
    len_diagonal = int(np.ceil(np.sqrt(width * width + height * height)))
    #rhos = np.linspace(-len_diagonal, len_diagonal, len_diagonal * 2.0)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    len_thetas = len(theta)

    print(len_diagonal)
    accumulator_circle = np.zeros((len_diagonal, len_diagonal), dtype=np.uint64)
    y_idx, x_idx = np.nonzero(image)
    print(len_thetas)

    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        for t_idx in range(len_thetas):
            a = int(round(x - radius * cos_theta[t_idx]) )
            b = int(round(y - radius * sin_theta[t_idx]) )
            #rho = int(round(x * cos_theta[t_idx] + y * sin_theta[t_idx]) + len_diagonal)
            #print(rho)
            accumulator_circle[a, b] += 1

    return accumulator_circle, theta


def hough_peaks_circle(H, num_peaks):
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


def displayCircles(peaks):
    for i in range(len(peaks)):
        
        cx = peaks[i][0]
        cy = peaks[i][1]
        #print(cx,cy)
        cv.circle(circle_org_image, (cx, cy), 20, (0, 255, 0), -1)

accumulator_circle, thetas_circle = generateHoughCircles(threshold_sobel_y )
cv.imwrite("accumulator_circle"+'.jpg',accumulator_circle)    

peaks_circle = hough_peaks_circle(accumulator_circle,150)

displayCircles(peaks_circle)

cv.imwrite("coin"+'.jpg',circle_org_image)
