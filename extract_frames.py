import cv2
import os

video_path = 'D:/python/model/yolov5/datasets/images/train/video.mp4'
output_dir = 'D:/python/model/yolov5/datasets/images/train'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 5  # Lấy 5 fps (frame mỗi giây), tùy chỉnh

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if int(cap.get(1)) % frame_rate == 0:
        cv2.imwrite(f'{output_dir}/frame_{count:04d}.jpg', frame)
        count += 1

cap.release()
print("Done!")
