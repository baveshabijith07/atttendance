import streamlit as st
import face_recognition
import cv2
import numpy as np
import pickle
import mysql.connector
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root1234",
    "database": "attendance_system",
}

def insert_attendance(roll_number):
    """ Insert attendance record into the database """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "INSERT INTO attendance (roll_number, timestamp) VALUES (%s, NOW())"
        cursor.execute(query, (roll_number,))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as err:
        st.error(f"❌ Database Error: {err}")
        return False

# Load trained model & label encoder
with open("face_recognition_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("known_face_encodings.pkl", "rb") as f:
    known_encodings = pickle.load(f)
with open("known_face_labels.pkl", "rb") as f:
    known_labels = pickle.load(f)

MODEL = "hog"  # 'hog' is faster, 'cnn' is more accurate but slower
TOLERANCE = 0.35  

def find_best_match(face_encoding):
    """ Find the best matching face using cosine similarity """
    similarities = cosine_similarity([face_encoding], known_encodings)
    best_match_idx = np.argmax(similarities)
    return known_labels[best_match_idx] if similarities[0][best_match_idx] > 0.6 else None

st.title("📸 AI-Based Attendance System")

if st.button("Capture & Verify Attendance"):
    # ✅ Fix: Reinitialize the camera every time
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # Small delay to allow the camera to initialize
    
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
    else:
        ret, frame = cap.read()
        
        if ret:
            # ✅ Fix: Ensure the frame updates properly
            st.image(frame, channels="BGR", caption="Captured Image")

            # ✅ Fix: Convert BGR to RGB BEFORE detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame)

            if face_encodings:
                recognized_roll = find_best_match(face_encodings[0])
                
                if recognized_roll:
                    if insert_attendance(recognized_roll):
                        st.success(f"✅ Attendance marked for {recognized_roll}")
                    else:
                        st.error("❌ Failed to save attendance. Check database connection.")
                else:
                    st.error("❌ No matching face found.")
            else:
                st.error("❌ No face detected in the image.")
        
        # ✅ Fix: Properly release the camera after each use
        cap.release()
        cv2.destroyAllWindows()
