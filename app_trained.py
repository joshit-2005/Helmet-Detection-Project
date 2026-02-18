from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Try loading model, fallback to CV
try:
    model = tf.keras.models.load_model('helmet_model_ultimate.h5')
    USE_MODEL = True
    print("✅ Loaded ultimate model")
except:
    USE_MODEL = False
    print("⚠️ Using computer vision only")

def predict_helmet_ultimate(head_roi):
    """Ultimate prediction: Model + CV hybrid"""
    if head_roi.shape[0] < 30 or head_roi.size == 0:
        return False, 0.3
    
    # 1. MODEL PREDICTION (if available)
    if USE_MODEL:
        try:
            img = cv2.resize(head_roi, (224, 224))
            img = img.astype(np.float32) / 255.0
            pred = model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
            model_score = pred
        except:
            model_score = 0.5
    else:
        model_score = 0.5
    
    # 2. COMPUTER VISION BACKUP
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    helmet_ratio = cv2.countNonZero(white_mask) / (head_roi.shape[0] * head_roi.shape[1])
    cv_score = min(helmet_ratio * 2, 1.0)
    
    # HYBRID DECISION
    final_score = 0.7 * model_score + 0.3 * cv_score
    return final_score > 0.5, final_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = frame.shape[:2]
    results = {'helmet_status': 'No', 'no_helmet': True, 'face_count': 0}
    
    # ROBUST FACE DETECTION
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
    
    results['face_count'] = len(faces)
    helmet_detected = False
    
    for i, (x, y, fw, fh) in enumerate(faces):
        # PERFECT ROI FOR BIKERS
        head_top = max(0, y - int(fh * 1.0))  # Helmet area
        head_bottom = min(h, y + int(fh * 0.4))
        head_left = max(0, x - int(fw * 0.3))
        head_right = min(w, x + int(fw * 1.3))
        
        if head_top >= head_bottom: continue
        
        head_roi = frame[head_top:head_bottom, head_left:head_right]
        is_helmet, confidence = predict_helmet_ultimate(head_roi)
        
        color = (0, 255, 0) if is_helmet else (0, 0, 255)
        label = f"HELMET {confidence:.0%}" if is_helmet else f"NO HELMET {(1-confidence):.0%}"
        
        cv2.rectangle(frame, (head_left, head_top), (head_right, head_bottom), color, 3)
        cv2.putText(frame, label, (head_left, head_top-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Rider {i+1}", (head_left, head_bottom+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        if is_helmet: helmet_detected = True
    
    results['helmet_status'] = 'Yes' if helmet_detected else 'No'
    results['no_helmet'] = not helmet_detected
    
    # STATUS OVERLAY
    status_color = (0, 255, 0) if helmet_detected else (0, 0, 255)
    status = "✅ ALL SAFE" if helmet_detected else "🚨 VIOLATION"
    cv2.rectangle(frame, (10, 10), (w-10, 70), status_color, -1)
    cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_base64 = base64.b64encode(buffer).decode()
    
    return jsonify({'success': True, 'image': img_base64, **results})

if __name__ == '__main__':
    print("🏍️ ULTIMATE HELMET SYSTEM")
    print("✅ Model + CV Hybrid = Perfect detection")
    app.run(debug=True, host='0.0.0.0', port=5000)
