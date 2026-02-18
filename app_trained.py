from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load TWO-STAGE model
try:
    model = tf.keras.models.load_model('head_helmet_model.h5')
    USE_MODEL = True
    print("✅ Loaded head-helmet model")
except:
    USE_MODEL = False
    print("⚠️ Using CV fallback - train first!")

def predict_helmet(head_roi):
    """TWO-STAGE: Head ROI → Helmet classifier"""
    if head_roi.shape[0] < 30 or head_roi.size == 0:
        return False, 0.3
    
    # DEEP LEARNING PREDICTION (80% weight)
    if USE_MODEL:
        try:
            img = cv2.resize(head_roi, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            pred = model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
            model_score = float(pred)
        except:
            model_score = 0.5
    else:
        model_score = 0.5
    
    # CV HELMET CONFIRMATION (20% weight)
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    helmet_ratio = cv2.countNonZero(white_mask) / (head_roi.shape[0] * head_roi.shape[1])
    cv_score = min(helmet_ratio * 2, 1.0)
    
    final_score = 0.8 * model_score + 0.2 * cv_score
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
    
    # ROBUST FACE DETECTION
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
    
    # TWO-STAGE LOGIC
    if len(faces) == 0:
        results = {
            'success': True, 
            'message': 'Invalid image - no person detected', 
            'helmet_status': 'None', 
            'no_helmet': False,
            'face_count': 0,
            'violations': 0
        }
        cv2.putText(frame, "NO PERSON DETECTED", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        helmet_detected = False
        violation_count = 0
        
        for i, (x, y, fw, fh) in enumerate(faces):
            # HEAD ROI (helmet area above + around face)
            head_top = max(0, y - int(fh * 1.2))
            head_bottom = min(h, y + int(fh * 0.3))
            head_left = max(0, x - int(fw * 0.2))
            head_right = min(w, x + int(fw * 1.2))
            
            if head_top >= head_bottom: continue
            
            head_roi = frame[head_top:head_bottom, head_left:head_right]
            is_helmet, confidence = predict_helmet(head_roi)
            
            # DRAW RESULTS
            color = (0, 255, 0) if is_helmet else (0, 0, 255)
            label = f"HELMET {confidence:.0%}" if is_helmet else f"NO HELMET {(1-confidence):.0%}"
            
            cv2.rectangle(frame, (head_left, head_top), (head_right, head_bottom), color, 3)
            cv2.putText(frame, label, (head_left, head_top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Rider {i+1}", (head_left, head_bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            if not is_helmet:
                violation_count += 1
            else:
                helmet_detected = True
        
        results = {
            'success': True,
            'helmet_status': 'Yes' if helmet_detected else 'No',
            'no_helmet': not helmet_detected,
            'face_count': len(faces),
            'violations': violation_count
        }
    
    # STATUS OVERLAY
    if 'message' in results:
        cv2.rectangle(frame, (10, 10), (w-10, 80), (0, 0, 255), -1)
        cv2.putText(frame, results['message'], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    elif helmet_detected:
        cv2.rectangle(frame, (10, 10), (w-10, 80), (0, 255, 0), -1)
        cv2.putText(frame, "ALL RIDERS SAFE ✓", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
    else:
        cv2.rectangle(frame, (10, 10), (w-10, 80), (0, 0, 255), -1)
        cv2.putText(frame, f"🚨 {results['violations']} VIOLATION(S)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_base64 = base64.b64encode(buffer).decode()
    
    return jsonify({**results, 'image': img_base64})

if __name__ == '__main__':
    print("🏍️ TWO-STAGE HELMET SYSTEM")
    print("1. Face Detection → 2. Head AI Analysis")
    app.run(debug=True, host='0.0.0.0', port=5000)
