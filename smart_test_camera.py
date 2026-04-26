import cv2
import time
import numpy as np

print("=" * 60)
print("SMART CAMERA DETECTOR")
print("Detecting REAL physical cameras vs Virtual/Dummy cameras")
print("=" * 60)

for idx in [0, 1, 2]:
    print(f"\nTesting Camera Index {idx}...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"  [X] Index {idx}: Could not open.")
        continue
    
    # Read a few frames to let it warm up
    for _ in range(5):
        cap.read()
        time.sleep(0.05)
        
    ret1, frame1 = cap.read()
    time.sleep(0.1)
    ret2, frame2 = cap.read()
    
    cap.release()
    
    if not ret1 or not ret2 or frame1 is None or frame2 is None:
        print(f"  [X] Index {idx}: Failed to read frames.")
        continue
        
    # Check 1: Is the image completely solid color? (Variance near 0)
    variance = np.var(frame1)
    
    # Check 2: Does the image have any live noise/motion?
    # A real camera will always have sensor noise between frames. A virtual still image will be exactly 0 difference.
    diff = cv2.absdiff(frame1, frame2)
    noise_level = np.mean(diff)
    
    print(f"  [-] Resolution: {frame1.shape}")
    print(f"  [-] Image Variance: {variance:.2f} (0 = solid color)")
    print(f"  [-] Inter-frame Noise: {noise_level:.4f} (0 = perfectly static/virtual)")
    
    if variance < 1.0:
        print(f"  [!] Index {idx} appears to be a DUMMY/BLANK virtual camera.")
    elif noise_level == 0.0:
        print(f"  [!] Index {idx} appears to be a STATIC IMAGE virtual camera.")
    else:
        print(f"  [SUCCESS] Index {idx} appears to be a REAL LIVE CAMERA! <<---- USE THIS ONE")

print("\n" + "=" * 60)
