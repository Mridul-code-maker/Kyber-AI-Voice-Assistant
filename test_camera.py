"""Quick camera diagnostic — run this standalone to check if your webcam is accessible."""
import cv2
import sys

print("=" * 60)
print("CAMERA DIAGNOSTIC")
print("=" * 60)
print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV build info (backends):")
for line in cv2.getBuildInformation().splitlines():
    line = line.strip()
    if "Video I/O" in line or "DirectShow" in line or "MSMF" in line or "DSHOW" in line:
        print(f"  {line}")
print()

backends = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("Default", None),
]

found_any = False

for idx in range(4):
    for backend_name, backend_flag in backends:
        try:
            if backend_flag is not None:
                cap = cv2.VideoCapture(idx, backend_flag)
            else:
                cap = cv2.VideoCapture(idx)

            opened = cap.isOpened()
            frame_ok = False
            frame_shape = None

            if opened:
                ret, frame = cap.read()
                frame_ok = ret and frame is not None
                if frame_ok:
                    frame_shape = frame.shape

            status = ""
            if not opened:
                status = "FAILED (isOpened=False)"
            elif not frame_ok:
                status = "PHANTOM (isOpened=True but read() failed — no real frames)"
            else:
                status = f"OK — Frame: {frame_shape}"
                found_any = True

            icon = "✓" if frame_ok else "✗"
            print(f"  [{icon}] Index {idx} ({backend_name:11s}): {status}")
            cap.release()

        except Exception as e:
            print(f"  [!] Index {idx} ({backend_name:11s}): EXCEPTION — {e}")

    print()

print("=" * 60)
if found_any:
    print("RESULT: At least one working camera was found.")
else:
    print("RESULT: NO working cameras found!")
    print()
    print("Possible causes:")
    print("  1. Another app is using the camera (Zoom, Teams, OBS, browser, etc.)")
    print("  2. Windows Camera Privacy is blocking Python")
    print("     → Settings > Privacy & Security > Camera > Let desktop apps access your camera")
    print("  3. Camera driver issue — try Device Manager > Cameras > right-click > Disable then Enable")
    print("  4. Antivirus software blocking camera access")
print("=" * 60)
