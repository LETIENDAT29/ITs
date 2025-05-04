import gradio as gr
import subprocess
import os
from PIL import Image

def detect_image(img):
    # Lưu ảnh tạm
    input_path = "input.jpg"
    img.save(input_path)

    # Xóa thư mục output cũ nếu cần
    if os.path.exists("runs/detect/exp"):
        import shutil
        shutil.rmtree("runs/detect/exp")

    # Gọi lệnh detect của YOLOv5
    subprocess.run([
        "python", "detect.py",
        "--weights", "runs/train/exp/weights/best.pt",  # Sửa đường dẫn nếu cần
        "--source", input_path,
        "--conf", "0.25"
    ])

    # Đường dẫn ảnh kết quả
    output_path = "runs/detect/exp/input.jpg"
    return Image.open(output_path)

# Giao diện Gradio
demo = gr.Interface(fn=detect_image, inputs="image", outputs="image", title="YOLOv5 Object Detection")

if __name__ == "__main__":
    demo.launch()
