import cv2 as cv
import os

#Start webcamera and saves images in greyscale to use for cameracalibration

# Folder to save images
save_folder = '' #Set filePath to folder to save images.
os.makedirs(save_folder, exist_ok=True)

# Open webcam (0 = default camera)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("📸 Press 'c' to capture an image when the chessboard is in view.")
print("🔚 Press 'q' to quit.")

i = 0
max_images = 10

while i < max_images:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Show live camera feed
    cv.imshow("Live Feed (Press 'c' to capture)", gray)

    key = cv.waitKey(1) & 0xFF
    if key == ord('c'):
        filename = os.path.join(save_folder, f'image_bw_{i+1:02d}.jpg')
        cv.imwrite(filename, gray)
        print(f"✅ Saved {filename}")
        i += 1
    elif key == ord('q'):
        print("👋 Quitting early.")
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
print("🎉 Done capturing grayscale images.")
