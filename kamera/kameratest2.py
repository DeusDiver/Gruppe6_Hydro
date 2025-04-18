import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from bisect import bisect_left

# ========== CONFIG ==========

# Camera source
builtIn = 0
usbCamera = 1

# Plant monitor settings
DECREASE_THRESHOLD = 10  # % decrease to trigger warning
GREEN_THRESHOLD = 10  # Absolute threshold for warning (in %)
MIN_DURATION = timedelta(seconds=30)  # Duration below threshold to trigger
TIME_WINDOW = timedelta(minutes=1)  # Trend window
READ_INTERVAL_SECONDS = 3  # How often to check plant

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "plant"

# ========== INIT ==========

os.makedirs('plantData', exist_ok=True)

cap = cv2.VideoCapture(builtIn)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

lower_green = np.array([30, 70, 70])
upper_green = np.array([80, 240, 240])

green_percentages = []
timestamps = []
below_threshold_start = None  # Track start time of threshold breach

def save_plant_data():
    filename = datetime.now().strftime("plantData/%Y-%m-%d_%H-%M-%S.csv")
    with open(filename, 'w') as f:
        f.write("timestamp,green_percentage\n")
        for ts, gp in zip(timestamps, green_percentages):
            f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{gp:.2f}\n")
    print(f"Data saved to {filename}")

print("🌿 Monitoring started. Press Ctrl+C to stop.")

try:
    while True:
        current_time = datetime.now()
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        green_percent = (np.count_nonzero(mask) / mask.size) * 100
        green_percentages.append(green_percent)
        timestamps.append(current_time)

        # Threshold-based warning check
        threshold_warning = 0
        if green_percent < GREEN_THRESHOLD:
            if below_threshold_start is None:
                below_threshold_start = current_time
            elapsed_time = current_time - below_threshold_start
            if elapsed_time >= MIN_DURATION:
                threshold_warning = 1
        else:
            below_threshold_start = None

        # Trend analysis (existing functionality)
        warning_message = None
        if len(timestamps) >= 2:
            window_start = current_time - TIME_WINDOW
            start_index = bisect_left(timestamps, window_start)
            recent_gps = green_percentages[start_index:]

            if len(recent_gps) >= 2:
                oldest_gp = recent_gps[0]
                newest_gp = recent_gps[-1]

                if oldest_gp != 0:
                    percent_change = ((newest_gp - oldest_gp) / oldest_gp) * 100
                    if percent_change <= -DECREASE_THRESHOLD:
                        mins = TIME_WINDOW.total_seconds() // 60
                        warning_message = f"WARNING: {abs(percent_change):.1f}% decrease in {mins} mins!"

        # Print status
        timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp_str}] Green: {green_percent:.2f}% | Warning: {threshold_warning}")
        if warning_message:
            print("🚨", warning_message)

        # Publish JSON payload
        payload = json.dumps({
            "green_percent": round(green_percent, 2),
            "warning": threshold_warning
        })
        client.publish(MQTT_TOPIC, payload)

        time.sleep(READ_INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\n⏹️ Monitoring stopped by user.")
    save_plant_data()

client.loop_stop()
client.disconnect()
cap.release()