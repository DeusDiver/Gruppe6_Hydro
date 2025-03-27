import cv2
import numpy as np
import os
from datetime import datetime

# Create plantData directory if it doesn't exist
os.makedirs('plantData', exist_ok=True)

# Open camera
cap = cv2.VideoCapture(1)  # 0 = Webkamera, 1 = første usbkamera os

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Green color range in HSV,   HSV = [HUE 0-180, Saturation 0-255, Brightness 0-255]
lower_green = np.array([30, 70, 70])
upper_green = np.array([80, 240, 240])

green_percentages = []
timestamps = []

def save_plant_data():
    """Save data to timestamped file in plantData folder"""
    filename = datetime.now().strftime("plantData/%Y-%m-%d_%H-%M-%S.csv")
    with open(filename, 'w') as f:
        f.write("timestamp,green_percentage\n")
        for ts, gp in zip(timestamps, green_percentages):
            f.write(f"{ts},{gp:.2f}\n")
    print(f"Data saved to {filename}")

while True:
    ret, frame = cap.read()  # Fixed indentation here
    if not ret:
        break

    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Image processing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate green percentage
    green_percent = (np.count_nonzero(mask) / mask.size) * 100
    green_percentages.append(green_percent)
    timestamps.append(current_time)
    
    # Display info
    cv2.putText(frame, f"{current_time}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Green: {green_percent:.2f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.imshow("Plant Monitoring", frame)
    cv2.imshow("Green Mask", mask)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        save_plant_data()
        break
    elif key & 0xFF == ord('s'):  # Manual save with 's' key
        save_plant_data()

cap.release()
cv2.destroyAllWindows()