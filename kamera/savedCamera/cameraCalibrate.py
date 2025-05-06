import numpy as np
import cv2 as cv
import glob
import os

# termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points for a 9x7 inner corner grid
objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Update path to your folder
image_folder = '' #Path to imagefolder for images saved from cameraCalibrationImagePrep.py
images = glob.glob(os.path.join(image_folder, '*.jpg'))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 7), None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9, 7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration data
np.savez('', #Path to folder to save calibration data.
         cameraMatrix=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("✅ Calibration data saved to 'calibration_data.npz'.")

# Load a test image and undistort it
img = cv.imread('') #Path to test image to undistort
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.0, (w, h))

# Undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)



# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv.imwrite('', dst) #Set Path, then get an image that is  calibrated.

# Calculate total reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("📏 Total reprojection error: {}".format(mean_error / len(objpoints)))

#Load and print calibration data
data = np.load('calibration_data.npz') #Load path to calibration data to check. 
cameraMatrix = data['cameraMatrix']
dist = data['dist']
rvecs = data['rvecs']
tvecs = data['tvecs']

print("📷 Camera Matrix:\n", cameraMatrix)
print("🎯 Distortion Coefficients:\n", dist)
