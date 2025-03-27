import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque

# Create data directory
os.makedirs('plantData', exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)        # Disable auto white balance
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Manual exposure mode
cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # Initial exposure value (adjust as needed)

# Iceberg lettuce parameters
lower_green = np.array([35, 40, 80])    # Hue, Saturation, Value
upper_green = np.array([75, 180, 220])
KERNEL_SIZE = 7                         # Morphology kernel size
BUFFER_SIZE = 15                        # Temporal median buffer
ROI_SCALE = 0.6                         # Center region of interest

# Initialize CLAHE for lighting normalization
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

# Data storage
percentage_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = []
green_percentages = []

def save_plant_data():
    """Save data with timestamp"""
    filename = datetime.now().strftime("plantData/%Y-%m-%d_%H-%M-%S.csv")
    with open(filename, 'w') as f:
        f.write("timestamp,green_percentage\n")
        for ts, gp in zip(timestamps, green_percentages):
            f.write(f"{ts},{gp:.2f}\n")
    print(f"Data saved to {filename}")

def process_lettuce_frame(frame):
    """Iceberg-specific processing pipeline"""
    # Lighting normalization in LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Noise reduction with bilateral filter
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # HSV thresholding
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get ROI (center 60%)
    h, w = frame.shape[:2]
    roi = frame[
        int(h*(0.5-ROI_SCALE/2)):int(h*(0.5+ROI_SCALE/2)),
        int(w*(0.5-ROI_SCALE/2)):int(w*(0.5+ROI_SCALE/2))
    ]
    
    # Process frame
    mask = process_lettuce_frame(roi)
    current_percent = (np.count_nonzero(mask) / mask.size * 100)
    
    # Temporal median filtering
    percentage_buffer.append(current_percent)
    final_percent = np.median(list(percentage_buffer))
    
    # Store data
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamps.append(current_time)
    green_percentages.append(final_percent)
    
    # Display
    display_frame = roi.copy()
    cv2.putText(display_frame, f"{current_time}", (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.putText(display_frame, f"Green: {final_percent:.1f}%", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    
    cv2.imshow("Iceberg Lettuce Monitor", display_frame)
    cv2.imshow("Plant Mask", mask)
    
    # Handle keys
    key = cv2.waitKey(1)
    if key == ord('q'):
        save_plant_data()
        break
    elif key == ord('s'):
        save_plant_data()

cap.release()
cv2.destroyAllWindows()