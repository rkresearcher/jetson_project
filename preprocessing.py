import matplotlib.pyplot as plt
import numpy as np
import cv2

# The input image.
cap = cv2.VideoCapture("Wavepool Lifeguard Rescue 58 - Spot the Drowning!-jz1qXpZotbc.mp4")
while (cap.isOpened()):
	ret, image = cap.read()
	params = cv2.SimpleBlobDetector_Params()
	bl = cv2.blur(image,(3,3))
	canny = cv2.Canny(bl,50,200)
	pts = np.argwhere(canny>0)
	y1,x1 = pts.min(axis=0)
	y2,x2 = pts.max(axis=0)
	n,m,t = image.shape
	print (n)
	image = image[10:m-50,10:n-40]
# Define thresholds
#Can define thresholdStep. See documentation. 
	params.minThreshold = 10
	params.maxThreshold = 200

# Filter by Area.
	params.filterByArea = True
	params.minArea = 50
	params.maxArea = 10000

# Filter by Color (black=0)
	params.filterByColor = False  #Set true for cast_iron as we'll be detecting black regions
	params.blobColor = 0

# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.5
	params.maxCircularity = 1

# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.5
	params.maxConvexity = 1

# Filter by InertiaRatio
	params.filterByInertia = True
	params.minInertiaRatio = 0
	params.maxInertiaRatio = 1

# Distance Between Blobs
	params.minDistBetweenBlobs = 0

# Setup the detector with parameters
	detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
	keypoints = detector.detect(image)

	print("Number of blobs detected are : ", len(keypoints))


# Draw blobs
	img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#plt.imshow(img_with_blobs)
	if ret == True:
		cv2.imshow("Keypoints", img_with_blobs)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()

# Save result
#cv2.imwrite("particle_blobs.jpg", img_with_blobs)
