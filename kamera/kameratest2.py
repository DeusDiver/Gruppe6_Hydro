import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from bisect import bisect_left

# Create plantData directory if it doesn't exist
os.makedirs('plantData', exist_ok=True)

builtIn = 0
usbCamera = 1

# Open camera
cap = cv2.VideoCapture(builtIn)  # NB!! CHOOSE CORRECTLY!!! If there are more cameras try 2,3....

# Configuration parameters
DECREASE_THRESHOLD = 10  # 10% decrease to trigger warning
TIME_WINDOW = timedelta(minutes=1)  # Monitoring window

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Green color range in HSV
lower_green = np.array([30, 70, 70])
upper_green = np.array([80, 240, 240])

green_percentages = []
timestamps = []  # Now stores datetime objects

def save_plant_data():
    """Save data to timestamped file in plantData folder"""
    filename = datetime.now().strftime("plantData/%Y-%m-%d_%H-%M-%S.csv")
    with open(filename, 'w') as f:
        f.write("timestamp,green_percentage\n")
        for ts, gp in zip(timestamps, green_percentages):
            f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{gp:.2f}\n")
    print(f"Data saved to {filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.now()
    
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
    
    # Trend analysis
    warning_message = None
    if len(timestamps) >= 2:
        # Find data points within our time window
        window_start = current_time - TIME_WINDOW
        start_index = bisect_left(timestamps, window_start)
        recent_gps = green_percentages[start_index:]
        
        if len(recent_gps) >= 2:
            oldest_gp = recent_gps[0]
            newest_gp = recent_gps[-1]
            
            if oldest_gp != 0:  # Prevent division by zero
                percent_change = ((newest_gp - oldest_gp) / oldest_gp) * 100
                if percent_change <= -DECREASE_THRESHOLD:
                    mins = TIME_WINDOW.total_seconds() // 60
                    warning_message = f"WARNING: {abs(percent_change):.1f}% decrease in {mins} mins!"

    # Display info
    display_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, display_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(frame, f"Green: {green_percent:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    if warning_message:
        cv2.putText(frame, warning_message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    cv2.putText(frame, "Press 'q' to exit, 's' to save", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.imshow("Plant Monitoring", frame)
    cv2.imshow("Green Mask", mask)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        save_plant_data()
        break
    elif key == ord('s'):
        save_plant_data()

cap.release()
cv2.destroyAllWindows()