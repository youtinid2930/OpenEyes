import cv2
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from tkinter import Label
from PIL import Image, ImageTk

mixer.init()
mixer.music.load("music.wav")
# Function to calculate the EAR
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate the MAR
def mouth_aspect_ratio(mouth):
    # A = distance between the left and right corners of the mouth
    A = distance.euclidean(mouth[1], mouth[7])
    
    # B = distance between the upper left and upper right points of the mouth
    B = distance.euclidean(mouth[2], mouth[6])
    
    # C = distance between the top center and bottom center of the mouth
    C = distance.euclidean(mouth[3], mouth[5])

    D = distance.euclidean(mouth[0], mouth[4])
    
    # The Mouth Aspect Ratio (MAR) formula is typically the ratio of the distances
    mar = (A + B+ C) / (3.0 * D)
    
    return mar

# Function to  strat Deteciton
def start_detection():
    global cap, running
    running = True
    startButton.pack_forget()
    stopButton.place(x=100, y=550)
    btn_test.place(x=250, y=550)
    cap = cv2.VideoCapture(0)
    detect()

# Function to stop Detection
def stop_detection():
    global running, cap
    running = False
    cap.release()

# Function to Display the warning of Yawning and being Tired
def ifTired(mar, frame):
    global yawn_count, yawn_thresh, yawn_flag
    if mar > yawn_thresh:
        yawn_flag += 1
        if yawn_flag >= yawn_frame_check:
            yawn_count += 1
            yawn_flag = 0
            cv2.putText(frame, "YAWNING DETECTED!", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        yawn_flag = 0
    if yawn_count >= yawn_limit:
        cv2.putText(frame, "YOU ARE TIRED!", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# Funciton to Dispaly Warning if there is a Drawnsiness and the person Sleep
def ifDrawsiness(ear, frame):
    global thresh, flag
    if ear < thresh:
        flag += 1
        if flag >= frame_check:
            cv2.putText(frame, "NA3SSSS!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()
    else:
        flag = 0

# Function to get left eye Cordonates
def getLeftEyeCordonates(landmarks, w, h):
    left_eye = [
        (int(landmarks[362].x * w), int(landmarks[362].y * h)),
        (int(landmarks[385].x * w), int(landmarks[385].y * h)),
        (int(landmarks[387].x * w), int(landmarks[387].y * h)),
        (int(landmarks[263].x * w), int(landmarks[263].y * h)),
        (int(landmarks[373].x * w), int(landmarks[373].y * h)),
        (int(landmarks[380].x * w), int(landmarks[380].y * h))
    ]
    return left_eye
# Function to get right eye Cordonates
def getRightEyeCordonates(landmarks, w, h):
    right_eye = [
        (int(landmarks[33].x * w), int(landmarks[33].y * h)),
        (int(landmarks[160].x * w), int(landmarks[160].y * h)),
        (int(landmarks[158].x * w), int(landmarks[158].y * h)),
        (int(landmarks[133].x * w), int(landmarks[133].y * h)),
        (int(landmarks[153].x * w), int(landmarks[153].y * h)),
        (int(landmarks[144].x * w), int(landmarks[144].y * h))
    ]
    return right_eye

# Function to get mouth cordonates
def getMouthCordonates(landmarks, w, h):
    mouth = [
        (int(landmarks[61].x * w), int(landmarks[61].y * h)),  # Left corner
        (int(landmarks[39].x * w), int(landmarks[39].y * h)),  # Upper left
        (int(landmarks[0].x * w), int(landmarks[0].y * h)),  # Top center
        (int(landmarks[269].x * w), int(landmarks[269].y * h)),# Upper right
        (int(landmarks[287].x * w), int(landmarks[287].y * h)),# Right corner
        (int(landmarks[405].x * w), int(landmarks[405].y * h)),  # Bottom center
        (int(landmarks[17].x * w), int(landmarks[17].y * h)),  # Bottom center
        (int(landmarks[181].x * w), int(landmarks[181].y * h)),  # Bottom center
    ]
    return mouth
# Function to get the mouth cordonates
def getCordonates(landmarks, w, h):
    left_eye = getLeftEyeCordonates(landmarks, w, h)
    right_eye = getRightEyeCordonates(landmarks, w, h)
    mouth = getMouthCordonates(landmarks, w, h)
    return left_eye, right_eye, mouth 

# Function to Draw contours
def drawContour(frame, area):
    cv2.polylines(frame, [np.array(area, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

# Fucntion Calculer le taux de fatigue par le nombre de clairement
def tauxFatigueByClairement(nombreClairement):
    if nombreClairement < 10:
        return 10  # Fatigue légère constante
    elif 10 <= nombreClairement <= 20:
        return 0  # Pas  fatigue
    elif 20 < nombreClairement <= 30:
        return 15 + (nombreClairement - 20) * (10 / 10)  # Augmentation linéaire
    elif nombreClairement > 30:
        return 25  # Fatigue maximale
    else:
        return 0

# Functions calcule taux de fatigue par le nombre de bâillement
def tauxFatigueByBaillement( numberBaillemnt):
        if numberBaillemnt == 0:
             return 0  # Pas de fatigue
        elif 1 <= numberBaillemnt <= 3:
             return 10 + 10 * (numberBaillemnt - 1)  # Fatigue légère
        elif 4 <= numberBaillemnt <= 6:
              return 30 + 10 * (numberBaillemnt - 4)  # Fatigue modérée
        elif numberBaillemnt > 6:
              return 50  # Fatigue avancée
        else:
              return 0

# Function calcule taux fatigue par clin
def tauxFatigueByClin(clins):
    
    duree_moyenne=sum(clins)/(len(clins))

    if duree_moyenne < 0.3:
        return 0  # Pas de fatigue
    elif 0.3 <= duree_moyenne <= 0.5:
        return 50 * (duree_moyenne - 0.3) / 0.2  # Fatigue croissante
    elif duree_moyenne > 0.5:
        return 50  # Fatigue avancée
    else:
        return 0

# Functions calcule taux fatigue avec nombre de headDrop
def tauxFatiguebyHeadDrop(nbre_headDrop):
       if nbre_headDrop == 0:
              return 0  # Pas de fatigue
       elif 1 <= nbre_headDrop <= 3:
              return 20 * nbre_headDrop  # Fatigue modérée
       elif nbre_headDrop > 3:
              return 100  # Fatigue critique
       else:
              return 0


def TauxFatigue(nbre_clainement,nbre_bâillement,clins,nbre_headDrop):
    
    tauxClairement = tauxFatigueByClairement(nbre_clainement)
    tauxBaillement = tauxFatigueByBaillement(nbre_bâillement)
    tauxClin = tauxFatigueByClin(clins)
    tauxHeadDrop = tauxFatiguebyHeadDrop(nbre_headDrop)
    
    return tauxClairement+tauxBaillement+tauxClin+tauxHeadDrop


# Detection logic
def detect():
    global running
    if not running:
        return

    ret, frame = cap.read()
    if ret:
        height, width, channels = frame.shape  # Get the dimensions
        print(f"Width: {width}, Height: {height}, Channels: {channels}")
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        global flag, yawn_flag, yawn_count, root
        
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape
            # Get the cordonates
            left_eye, right_eye, mouth = getCordonates(landmarks, w, h)
            # Get and calculate the Eyes Aspect Ration (EAR)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            # Get the Mouth Aspect Ratio (MAR)
            mar = mouth_aspect_ratio(mouth)
            
                
            
            if draw_contours:
                drawContour(frame, left_eye)
                drawContour(frame, right_eye)
            drawContour(frame, mouth)
            # display sleep warning by eyes
            ifDrawsiness(ear, frame)
            # display warnings if tired
            ifTired(mar, frame)
            
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    videoLabel.imgtk = imgtk
    videoLabel.configure(image=imgtk)
    videoLabel.after(10, detect)

# Function to Controlle the Draw of Contours
def controlleDrawContour():
    global draw_contours
    draw_contours = not draw_contours
# Thresholds and parameters
thresh = 0.25
yawn_thresh = 0.5
frame_check = 20
yawn_frame_check = 15
yawn_limit = 3
flag = 0
yawn_flag = 0
yawn_count = 0
buttonCreated = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils
drawing_spec = drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

cap = None
running = False

draw_contours = False

root = tk.Tk()
root.title("OpenEyes")
root.geometry("640x640")
root.resizable(False, False)

root.configure(bg="#121212")  # Dark background





background_image = Image.open("./asset/image-640x640.png")  # Replace with your image file
  # Resize to fixed window size
bg_image = ImageTk.PhotoImage(background_image)

# Create a label to hold the background image
bg_label = tk.Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1) 


# Styles


button_style = {
    "bg": "#1E88E5",  # Blue background
    "fg": "white",  # White text
    "activebackground": "#1565C0",  # Darker blue on hover
    "activeforeground": "white",  # White text on hover
    "font": ("Helvetica", 9, "bold"),
    "bd": 0,  # Borderless
    "relief": "flat",
    "width": 20,
    "height": 2
}

label_style = {
    "bg": "#1E88E5",  # Match the background
    "fg": "white",  # White text
    "font": ("Helvetica", 14, "bold")
}

# Video Label
videoLabel = Label(root, text="Keeping Drivers Awake, One Blink at a Time", **label_style)
videoLabel.pack(pady=20)

# Buttons
startButton = tk.Button(root, text="Start Detection", command=start_detection, **button_style)
startButton.pack(pady=10)

stopButton = tk.Button(root, text="Stop Detection", command=stop_detection, **button_style)
stopButton.pack_forget()  # Initially hidden

btn_test = tk.Button(root, text="Draw Eyes", command=controlleDrawContour, **button_style)
btn_test.pack_forget()  # Initially hidden



# Main Loop
root.mainloop()