from ultralytics import YOLO
import cv2
import math 
import json

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

data = json.load(open('params.json'))
PIXEL_SIZE = data['pixel_size']/1000
FOCAL_LEN = data['focal_length']/1000
REAL_HEIGHT = data['real_height']/1000

if data['model'].lower() == 'medium':
    model = YOLO('YOLOv10m_sign.pt')
else:
    model = YOLO('YOLOv10n_sign.pt')

'''
if data['model'].lower() == 'yolov10m':
    try:
        model = YOLO('YOLOv10m_sign.onnx')
    except:
        model = YOLO('YOLOv10m_sign.pt')
        model.export(format = 'onnx')
        model = YOLO('YOLOv10m_sign.onnx')
'''
class_names = ['Pedestrian Crossing', 'Equal-level Intersection', 'No Entry', 'Right Turn Only', 'Intersection', 'Intersection with Uncontrolled Road', 'Dangerous Turn', 'No Left Turn', 'Bus Stop', 'Roundabout', 'No Stopping and No Parking', 'U-Turn Allowed', 'Lane Allocation', 'No Left Turn for Motorcycles', 'Slow Down', 'No Trucks Allowed', 'Narrow Road on the Right', 'No Passenger Cars and Trucks', 'Height Limit', 'No U-Turn', 'No U-Turn and No Right Turn', 'No Cars Allowed', 'Narrow Road on the Left', 'Uneven Road', 'No Two or Three-wheeled Vehicles', 'Customs Checkpoint', 'Motorcycles Only', 'Obstacle on the Road', 'Children Present', 'Trucks and Containers', 'No Motorcycles Allowed', 'Trucks Only', 'Road with Surveillance Camera', 'No Right Turn', 'Series of Dangerous Turns', 'No Containers Allowed', 'No Left or Right Turn', 'No Straight and Right Turn', 'Intersection with T-Junction', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (80km/h)', 'Speed limit (40km/h)', 'Left Turn', 'Low Clearance', 'Other Danger', 'Go Straight', 'No Parking', 'Containers Only', 'No U-Turn for Cars', 'Level Crossing with Barriers']


def distance(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
    object_height = (abs(x2 - x1) + abs(y2 - y1))/2 
    # print(FOCAL_LEN*REAL_HEIGHT*)
    return FOCAL_LEN*REAL_HEIGHT/object_height/PIXEL_SIZE

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            d = distance(box)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(img, class_names[cls] + ' (' + str(round(d, 3)) + ' m)', org, font, font_scale, color, thickness)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()