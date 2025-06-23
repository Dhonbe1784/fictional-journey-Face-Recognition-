import os
import cv2
import numpy as np

# Configuration (adjust for performance)
DATASET_PATH = 'faces_dataset'
TRAINED_MODEL = 'face_recognizer.yml'
MIN_FACE_SIZE = (100, 100)  # Smaller faces for low-res cameras
RECOGNITION_THRESHOLD = 70   # Lower = more strict (adjust based on your needs)
RESIZE_SCALE = 0.5           # Reduce processing resolution
SKIP_FRAMES = 2              # Process every nth frame

# Create required directories
os.makedirs(DATASET_PATH, exist_ok=True)

# Initialize face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def capture_face_samples():
    """Capture face samples for training"""
    person_name = input("Enter person's name: ").strip()
    person_dir = os.path.join(DATASET_PATH, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cam = cv2.VideoCapture(0)
    print(f"Capturing samples for {person_name}. Press 'q' to stop...")
    count = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Downscale for faster processing
        small_frame = cv2.resize(frame, (0,0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, 
                         (int(x/RESIZE_SCALE), int(y/RESIZE_SCALE)),
                         (int((x+w)/RESIZE_SCALE), int((y+h)/RESIZE_SCALE)),
                         (0, 255, 0), 2)
            
            # Save face ROI
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)
            count += 1
        
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:  # Capture 50 samples
            break
    
    cam.release()
    cv2.destroyAllWindows()

def train_model():
    """Train the face recognition model"""
    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    for root, _, files in os.walk(DATASET_PATH):
        if os.path.basename(root) == DATASET_PATH:
            continue
            
        # Get label name
        name = os.path.basename(root)
        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1
            
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                
                # Resize for consistent training
                img = cv2.resize(img, MIN_FACE_SIZE)
                
                faces.append(img)
                labels.append(label_ids[name])
    
    if len(faces) == 0:
        print("No training data found!")
        return
    
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINED_MODEL)
    print(f"Trained on {len(faces)} samples. Model saved to {TRAINED_MODEL}")

def recognize_faces():
    """Run real-time face recognition"""
    if not os.path.exists(TRAINED_MODEL):
        print("First train the model!")
        return
    
    recognizer.read(TRAINED_MODEL)
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue
        
        # Downscale processing frame
        small_frame = cv2.resize(frame, (0,0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )
        
        for (x, y, w, h) in faces:
            # Recognize face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, MIN_FACE_SIZE)
            
            label_id, confidence = recognizer.predict(face_roi)
            
            # Scale coordinates back to original frame
            x_orig = int(x / RESIZE_SCALE)
            y_orig = int(y / RESIZE_SCALE)
            w_orig = int(w / RESIZE_SCALE)
            h_orig = int(h / RESIZE_SCALE)
            
            # Get label name
            label_name = "Unknown"
            for name, id in label_ids.items():
                if id == label_id and confidence < RECOGNITION_THRESHOLD:
                    label_name = name
                    break
            
            # Draw results
            color = (0, 255, 0) if label_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x_orig, y_orig), 
                         (x_orig + w_orig, y_orig + h_orig), 
                         color, 2)
            cv2.putText(frame, f"{label_name} {confidence:.1f}", 
                       (x_orig, y_orig - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Load label mapping
label_ids = {}
def load_labels():
    global label_ids
    for i, name in enumerate(os.listdir(DATASET_PATH)):
        if os.path.isdir(os.path.join(DATASET_PATH, name)):
            label_ids[name] = i

# Main menu
if __name__ == "__main__":
    load_labels()
    
    while True:
        print("\nFace Recognition System")
        print("1. Capture new face samples")
        print("2. Train model")
        print("3. Run face recognition")
        print("4. Exit")
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            capture_face_samples()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            break
        else:
            print("Invalid choice!")

print("System exited")
