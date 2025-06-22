import torch
import gradio as gr
from PIL import Image
import numpy as np

# Load mô hình YOLOv5 từ Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Sử dụng GPU nếu có
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Hàm xử lý ảnh đầu vào
def detect_objects(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    results = model(image)
    results.render()

    # Trích xuất nhãn
    labels = results.names
    pred_classes = results.pred[0][:, -1].tolist() if results.pred[0].shape[0] > 0 else []
    label_names = [labels[int(cls)] for cls in pred_classes]
    unique_labels = sorted(set(label_names))
    label_text = ", ".join(unique_labels) if unique_labels else "Không phát hiện vật thể nào."

    return results.ims[0], label_text

# Giao diện Gradio với Submit và Clear
with gr.Blocks(title="Phát hiện vật thể với YOLOv5") as demo:
    gr.Markdown("## <center>🧠 Phát hiện vật thể với YOLOv5s</center>")
    gr.Markdown("Tải ảnh lên để phát hiện vật thể bằng mô hình **YOLOv5s** của UTH.")

    with gr.Row():
        # Bố cục chia thành 2 cột
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="📤 Ảnh đầu vào")
            
            # Nút Clear và Submit dưới ảnh đầu vào bên trái
            submit_btn = gr.Button("🚀 Submit")
            clear_btn = gr.Button("🧹 Clear")
        
        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="📸 Ảnh kết quả")
            labels_out = gr.Textbox(label="🏷️ Danh sách nhãn phát hiện", interactive=False)
    
    submit_btn.click(fn=detect_objects, inputs=input_img, outputs=[output_img, labels_out])
    clear_btn.click(
        fn=lambda: (None, None, ""), 
        inputs=[], 
        outputs=[input_img, output_img, labels_out]
    )

if __name__ == "__main__":
    demo.launch(share=False)
