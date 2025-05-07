import cv2 as cv
import numpy as np

#Compares distorted and undistorted camerafeed.
#Check https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html for more information

# Load calibration data
data = np.load('') #Set path to load relevant calibration_data.npz.
cameraMatrix = data['cameraMatrix']
dist = data['dist']

# Open webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Get frame size from camera
ret, frame = cap.read()
if not ret:
    print("⚠️ Failed to grab initial frame")
    cap.release()
    exit()

h, w = frame.shape[:2]

# Compute optimal new camera matrix (alpha=1.0 = no cropping, described more in the tutorial from opencv)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1.0, (w, h))

print("📷 Starting undistorted + cropped camera feed (press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Undistort frame
    dst = cv.undistort(frame, cameraMatrix, dist, None, newcameramtx)

    # Crop using ROI
    x, y, w, h = roi
    dst_cropped = dst[y:y + h, x:x + w]

    # Resize cropped image to match original for side-by-side display
    dst_resized = cv.resize(dst_cropped, (frame.shape[1], frame.shape[0]))

    # Show original and undistorted cropped image side by side
    combined = np.hstack((frame, dst_resized))
    cv.imshow('Original (Left) | Undistorted + Cropped (Right)', combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
print("🛑 Camera feed closed.")
