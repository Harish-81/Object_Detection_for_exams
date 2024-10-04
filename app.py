import os
import cv2
import torch
from flask import Flask, render_template, Response

app = Flask(__name__)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  


TARGET_CLASSES = ['book', 'cell phone', 'laptop']


def detect_objects(frame):
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    for detection in results.xyxy[0]: 
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = model.names[int(class_id)]
        if class_name in TARGET_CLASSES:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return frame


def gen_frames():
    cap = cv2.VideoCapture(0)  
    while True:
        success, frame = cap.read() 
        if not success:
            break
        else:
            frame = detect_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
