import json
import webbrowser
from flask import Flask, request, Response
from flask_cors import CORS
from ultralytics import YOLO
from waitress import serve
from PIL import Image

app = Flask(__name__)
CORS(app)  # Bật CORS cho toàn bộ ứng dụng

class_names = ['Vach qua duong cho nguoi di bo', 'Nga ba bang nhau', 'Cam vao', 'Chi re phai', 'Nga tu', 'Nga tu khong kiem soat', 'Re nguy hiem', 'Cam re trai', 'Ben xe buyt', 'Vong xuyen', 'Cam dung va do xe', 'Cho phep quay dau', 'Phan lan', 'Cam re trai doi voi xe may', 'Giam toc do', 'Cam xe tai', 'Duong hep ben phai', 'Cam xe khach va xe tai', 'Gioi han chieu cao', 'Cam quay dau', 'Cam quay dau va re phai', 'Cam xe hoi', 'Duong hep ben trai', 'Duong khong bang phang', 'Cam xe hai hoac ba banh', 'Diem kiem tra hai quan', 'Chi xe may', 'Chuong ngai vat tren duong', 'Co tre em', 'Xe tai va container', 'Cam xe may', 'Chi xe tai', 'Duong co camera giam sat', 'Cam re phai', 'Choi cac khuc re nguy hiem', 'Cam container', 'Cam re trai hoac phai', 'Cam di thang va re phai', 'Nga tu voi nga ba T', 'Gioi han toc do (50km/h)', 'Gioi han toc do (60km/h)', 'Gioi han toc do (80km/h)', 'Gioi han toc do (40km/h)', 'Re trai', 'Chieu cao thap', 'Nguy hiem khac', 'Di thang', 'Cam do xe', 'Chi container', 'Cam quay dau doi voi xe hoi', 'Giao cat co rao chan']

data = json.load(open('params.json'))
PIXEL_SIZE = data['pixel_size']/1000
FOCAL_LEN = data['focal_length']/1000
REAL_HEIGHT = data['real_height']/1000


if data['model'].lower() == 'medium':
    model = YOLO('YOLOv10m_sign.pt')
else:
    model = YOLO('YOLOv10n_sign.pt')

def distance(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
    object_height = (abs(x2 - x1) + abs(y2 - y1))/2 
    # print(FOCAL_LEN*REAL_HEIGHT*)
    return FOCAL_LEN*REAL_HEIGHT/object_height/PIXEL_SIZE

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"]) # this function will be modified.
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
        json.dumps(boxes),  
        mimetype='application/json'
    )



def detect_objects_on_image(buf):
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        d = distance(box)
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, class_names[int(class_id)] + ' (' + str(round(d, 2)) + ' m)', prob
        ])
    return output


# webbrowser.open('http:///localhost:8080')  
serve(app, host='0.0.0.0', port=8080)
