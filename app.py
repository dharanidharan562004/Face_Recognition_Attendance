import cv2
import numpy as np
import os
import pickle
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import database

# --- FEATURE FLAGS ---
USE_FACE_RECOGNITION_LIB = False
try:
    import face_recognition # type: ignore
    from utils import liveness
    USE_FACE_RECOGNITION_LIB = True
    print("[INFO] face_recognition library loaded. Using Advanced Liveness & Recognition.")
except ImportError as e:
    print(f"[WARN] face_recognition/dlib not found ({e}). Falling back to OpenCV Haar Cascades.")
    print("[WARN] Liveness detection will be DISABLED in this mode.")
    
import email_service

app = Flask(__name__)
app.secret_key = "super_secret_key"

@app.route('/test_email')
def test_email_route():
    try:
        # Test Credentials
        import smtplib
        from email.mime.text import MIMEText
        
        sender = email_service.SENDER_EMAIL
        password = email_service.SENDER_PASSWORD
        
        if "your_email" in sender or "your_app_password" in password:
            return "ERROR: You have not configured email_service.py with your actual Email and App Password yet."
            
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.quit()
        return f"SUCCESS: Connection to Gmail established with {sender}. Application is ready to send emails."
    except Exception as e:
        return f"EMAIL CONNECTION FAILED: {str(e)}"

# Ensure directories exist
os.makedirs("database", exist_ok=True)
os.makedirs("static/faces", exist_ok=True)

# Global Variables
known_face_encodings = []
known_face_names = []
known_face_ids = []

# Registration State
registration_active = False
new_user_id = None
new_user_name = None

# Liveness State
blink_counter = 0
TOTAL_BLINKS = 0
liveness_verified = False

# Haar Cascade (Fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_faces():
    global known_face_encodings, known_face_names, known_face_ids
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []
    
    conn = database.sqlite3.connect(database.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_number FROM users")
    users = cursor.fetchall()
    
    for user in users:
        uid, name, roll = user
        path = f"static/faces/{uid}.npy"
        if os.path.exists(path):
            try:
                encoding = np.load(path)
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                known_face_ids.append(uid)
            except:
                pass
    
    print(f"Loaded {len(known_face_encodings)} faces.")
    conn.close()

# Initialize DB and Faces
database.init_db()
load_faces()

@app.route('/')
def index():
    return render_template('index.html', advanced_mode=USE_FACE_RECOGNITION_LIB)

@app.route('/logs')
def logs():
    data = database.get_logs()
    return render_template('logs.html', logs=data)

@app.route('/delete_log', methods=['POST'])
def delete_log_route():
    log_id = request.form.get('log_id')
    username = request.form.get('admin_user')
    password = request.form.get('admin_pass')
    
    if username == ADMIN_USER and password == ADMIN_PASS:
        database.delete_log(log_id)
        flash("Log deleted.", "info")
    else:
        flash("Error: Incorrect Admin Credentials!", "error")
        
    return redirect(url_for('logs'))

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/users')
def users_list():
    users = database.get_users()
    return render_template('users.html', users=users)

@app.route('/reports', methods=['GET', 'POST'])
def reports():
    date_filter = None
    if request.method == 'POST':
        date_filter = request.form.get('date')
    
    # If no filter, maybe default to today? Or all. Let's do all for now or user selection.
    if not date_filter and request.method == 'GET':
        import datetime
        date_filter = datetime.date.today().strftime("%Y-%m-%d") # Default to today
        
    report_data = database.get_attendance_report(date_filter)
    return render_template('reports.html', report=report_data, selected_date=date_filter)

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/register_student', methods=['POST'])
def register_student():
    global new_user_id, new_user_name, registration_active
    name = request.form.get('name')
    roll_number = request.form.get('roll_number')
    email = request.form.get('email')
    
    user_id = database.add_user(name, roll_number, email)
    if user_id:
        registration_active = True
        new_user_id = user_id
        new_user_name = name
        
        # Send Welcome Email
        if email:
             try:
                 email_service.send_registration_email(email, name, roll_number)
             except Exception as e:
                 print(f"Registration Email Failed: {e}")
                 
        flash(f"User {name} added! Please look at the camera to capture face.", "success")
        return render_template('register.html') 
    else:
        flash("Error: Roll number already exists!", "error")
        return redirect(url_for('register_page'))

# --- HELPER: FALLBACK RECOGNITION ---
known_face_images = [] # Store histograms or raw images
def get_face_histogram(image):
    # Convert to HSV for better color matching
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def load_faces_fallback():
    global known_face_images, known_face_names, known_face_ids
    known_face_images = []
    known_face_names = []
    known_face_ids = []
    
    conn = database.sqlite3.connect(database.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_number FROM users")
    users = cursor.fetchall()
    
    for user in users:
        uid, name, roll = user
        path = f"static/faces/{uid}.jpg" # Check for JPG in fallback
        if os.path.exists(path):
            try:
                img = cv2.imread(path)
                hist = get_face_histogram(img)
                known_face_images.append(hist)
                known_face_names.append(name)
                known_face_ids.append(uid)
            except:
                pass
    conn.close()

# --- LBPH MACHINE LEARNING ---
lbph_recognizer = None
if not USE_FACE_RECOGNITION_LIB:
    try:
        # Check if cv2 has face module (requires opencv-contrib-python)
        lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("[WARN] opencv-contrib-python not found. LBPH ML unavailable.")

def train_model():
    if not lbph_recognizer: return False
    
    faces = []
    ids = []
    
    # Iterate through faces folder
    # Assuming filenames are {id}.jpg
    face_dir = "static/faces"
    files = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
    
    if not files: return False
    
    for file in files:
        path = os.path.join(face_dir, file)
        try:
            # ID is the filename part before extension
            uid = int(os.path.splitext(file)[0])
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(uid)
        except ValueError:
            continue
            
    if len(faces) > 0:
        lbph_recognizer.train(faces, np.array(ids))
        lbph_recognizer.write('trainer.yml')
        print(f"[INFO] Model trained with {len(faces)} faces.")
        return True
    return False

def load_faces_fallback_forced():
    # Helper to clean reload faces
    if USE_FACE_RECOGNITION_LIB:
        load_faces()
    else:
        # Load ID map for displaying names
        global known_face_names, known_face_ids
        known_face_names = []
        known_face_ids = []
        conn = database.sqlite3.connect(database.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users")
        for uid, name in cursor.fetchall():
            known_face_names.append(name)
            known_face_ids.append(uid)
        conn.close()
        
        # Load Model
        if lbph_recognizer and os.path.exists('trainer.yml'):
            lbph_recognizer.read('trainer.yml')
            print("[INFO] LBPH Model Loaded.")

if not USE_FACE_RECOGNITION_LIB:
    load_faces_fallback_forced()

@app.route('/train')
def train_route():
    if train_model():
        flash("Machine Learning Model Trained Successfully!", "success")
    else:
        flash("Training Failed. Ensure you have registered users.", "error")
    return redirect(url_for('users_list'))

# Admin Credentials (Hardcoded for simplicity)
ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    user_id = request.form.get('user_id')
    username = request.form.get('admin_user')
    password = request.form.get('admin_pass')
    
    if username == ADMIN_USER and password == ADMIN_PASS:
        # 1. Delete from DB
        database.delete_user(user_id)
        
        # 2. Delete Face File
        extensions = ['.npy', '.jpg']
        for ext in extensions:
            path = f"static/faces/{user_id}{ext}"
            if os.path.exists(path):
                os.remove(path)
                
        # 3. Reload Faces Memory
        load_faces_fallback_forced()
        
        flash("User deleted successfully.", "info")
    else:
        flash("Error: Incorrect Admin Credentials!", "error")
        
    return redirect(url_for('users_list'))

# Global Welcome Message State
# {user_id: frames_remaining}
welcome_state = {}
registration_success_timer = 0 # Timer to show success message
current_attendance_mode = None # None = Auto, "IN", "OUT"

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_attendance_mode
    if mode in ['IN', 'OUT']:
        current_attendance_mode = mode
    else:
        current_attendance_mode = None
    return "OK"

def gen_frames():
    global registration_active, new_user_id, known_face_encodings, known_face_names
    global blink_counter, TOTAL_BLINKS, liveness_verified, welcome_state, registration_success_timer
    global current_attendance_mode
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Resize for performance 
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # --- SUCCESS MESSAGE OVERLAY ---
        if registration_success_timer > 0:
            cv2.rectangle(frame, (0,0), (640, 60), (0,255,0), -1)
            cv2.putText(frame, "REGISTRATION SUCCESS!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "Please Go & TRAIN MODEL", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            registration_success_timer -= 1
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue # Skip other checks
        
        # --- REGISTRATION LOGIC ---
        if registration_active:
            cv2.putText(frame, "REGISTRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if USE_FACE_RECOGNITION_LIB:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if len(face_locations) == 1:
                    face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
                    
                    # Check if already registered
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0 and (True in matches):
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            existing_name = known_face_names[best_match_index]
                            cv2.putText(frame, f"ALREADY REGISTERED: {existing_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        save_path = f"static/faces/{new_user_id}.npy"
                        np.save(save_path, face_encoding)
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(new_user_name)
                        known_face_ids.append(new_user_id)
                        registration_active = False
                        registration_success_timer = 60 # Trigger Success
                        cv2.putText(frame, "SUCCESS! REGISTERED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                 # Fallback Registration Logic
                 faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
                 if len(faces) == 1:
                    (x,y,w,h) = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    curr_hist = get_face_histogram(face_roi)
                    best_score = 0
                    for i, hist in enumerate(known_face_images):
                        score = cv2.compareHist(curr_hist, hist, cv2.HISTCMP_CORREL)
                        if score > best_score: best_score = score
                    
                    if best_score > 0.6:
                        cv2.putText(frame, "ALREADY REGISTERED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        save_path = f"static/faces/{new_user_id}.jpg"
                        cv2.imwrite(save_path, face_roi)
                        hist = get_face_histogram(face_roi)
                        known_face_images.append(hist)
                        known_face_names.append(new_user_name)
                        known_face_ids.append(new_user_id)
                        registration_active = False
                        registration_success_timer = 60 # Trigger Success
                        cv2.putText(frame, "REGISTERED (Basic Mode)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- ATTENDANCE MODE ---
        else:
            # Check Active Welcomes
            to_remove = []
            for uid, frames in welcome_state.items():
                if frames > 0:
                     # Find name for uid
                     if uid in known_face_ids:
                         idx = known_face_ids.index(uid)
                         name = known_face_names[idx]
                         cv2.putText(frame, f"WELCOME {name}!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                     welcome_state[uid] -= 1
                else:
                    to_remove.append(uid)
            for uid in to_remove:
                del welcome_state[uid]

            if USE_FACE_RECOGNITION_LIB:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

                for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    closed, ear = liveness.is_blinking(landmarks)
                    color = (0, 0, 255)
                    name = "Unknown"
                    status = "Fake/Photo?"
                    
                    if closed: blink_counter += 1
                    else:
                        if blink_counter >= 2: TOTAL_BLINKS += 1; liveness_verified = True
                        blink_counter = 0

                    if liveness_verified:
                        status = "Live"
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                                uid = known_face_ids[best_match_index]
                                color = (0, 255, 0)
                                marked = database.mark_attendance(uid)
                                if marked:
                                     welcome_state[uid] = 45 # Show for 45 frames
                                     status = "Marked!"
                                else:
                                    status = "Marked"

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, f"{name} ({status})", (left, bottom + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            else:
                # Fallback: LBPH Prediction
                faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
                for (x,y,w,h) in faces:
                    face_roi = gray_frame[y:y+h, x:x+w]
                    
                    best_name = "Unknown"
                    best_uid = None
                    confidence = 0
                    is_match = False
                    
                    if lbph_recognizer:
                        try:
                            uid_pred, conf = lbph_recognizer.predict(face_roi)
                            # LBPH Confidence: Lower is better. 0 is perfect match.
                            # Standard threshold is usually around 80-100.
                            if conf < 90: 
                                is_match = True
                                best_uid = uid_pred
                                # Find Name
                                if best_uid in known_face_ids:
                                    idx = known_face_ids.index(best_uid)
                                    best_name = known_face_names[idx]
                                confidence = int(max(0, 100 - conf))
                        except Exception as e:
                            print(f"Prediction Error: {e}")
                            pass
                    
                    color = (0, 165, 255) # Orange (Detecting)
                    
                    if is_match:
                        color = (0, 255, 0)
                        # Mark Attendance (Returns: success, type/msg)
                        # Use Global Manual Mode if set
                        success, status_type = database.mark_attendance(best_uid, force_status=current_attendance_mode)
                        
                        if success:
                             welcome_state[best_uid] = 45 # Show for 45 frames
                             status = f"{status_type} SUCCESS"
                             
                             # Send Email Notification
                             try:
                                 user_email = database.get_user_email(best_uid)
                                 if user_email:
                                     from datetime import datetime
                                     time_now = datetime.now().strftime("%I:%M %p")
                                     email_service.send_attendance_email(user_email, name, status_type, time_now)
                             except Exception as e:
                                 print(f"Email Error: {e}")
                                 
                        else:
                             status = f"{status_type}" # e.g. "Wait 1 min (IN)"
                    else:
                        status = "Unknown"
                        best_name = "Unknown"
                        
                    cv2.rectangle(frame,(x,y),(x+w,y+h), color, 2)
                    cv2.putText(frame, f"{best_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Global Status Info
                cv2.rectangle(frame, (0,0), (640, 40), (0,0,0), -1) 
                cv2.putText(frame, "ML MODE (LBPH)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
