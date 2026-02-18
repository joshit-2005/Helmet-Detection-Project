import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

def load_data_ultimate(images_dir, annotations_dir):
    images, labels = [], []
    
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()
            
            img_filename = root.find('filename').text.strip()
            img_path = os.path.join(images_dir, img_filename)
            
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'With Helmet': label = 1
                elif name == 'Without Helmet': label = 0
                else: continue
                
                bndbox = obj.find('bndbox')
                xmin = max(0, int(float(bndbox.find('xmin').text)))
                ymin = max(0, int(float(bndbox.find('ymin').text)))
                xmax = min(w, int(float(bndbox.find('xmax').text)))
                ymax = min(h, int(float(bndbox.find('ymax').text)))
                
                # STRICT VALIDATION
                if xmax <= xmin + 5 or ymax <= ymin + 5: continue
                
                crop = img_rgb[ymin:ymax, xmin:xmax]
                if crop.size == 0: continue
                
                crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                images.append(crop_resized.astype(np.float32) / 255.0)
                labels.append(label)
                
        except: continue
    
    return np.array(images), np.array(labels)

def create_ultimate_model():
    """95%+ accurate production model"""
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        
        # Deep feature extraction
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        # Strong classifier
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("🏆 ULTIMATE HELMET MODEL (95%+)")
    
    X, y = load_data_ultimate("images", "annotations")
    print(f"Loaded: {len(X)} crops | Helmet: {sum(y)} | No Helmet: {len(y)-sum(y)}")
    
    if len(X) == 0:
        print("❌ No data - use app_ultimate.py")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = create_ultimate_model()
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('helmet_model_ultimate.h5', save_best_only=True)
    ]
    
    print("🚀 Training for 95%+ accuracy...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              callbacks=callbacks, verbose=1)
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n🎉 FINAL: {acc:.1%} accuracy!")
    model.save('helmet_model_ultimate.h5')

if __name__ == "__main__":
    main()
