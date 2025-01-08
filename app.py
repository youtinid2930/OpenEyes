import cv2
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time
import threading
import asyncio

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
    global cap, running, globalTime
    running = True
    globalTime = time.time()
    # pack or forget the button in the window
    startButton.pack_forget()
    stopButton.place(x=120, y=550)
    drawEyesButton.place(x=220, y=550)
    drawMouthButton.place(x=320, y=550)
    drawFaceLandmarksButton.place(x=420, y=550)

    cap = cv2.VideoCapture(0)

    # Get FPS from the capture object
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    
    # Calculate the number of frames in 1 second
    frames_in_1_second = int(fps)
    print(f"Number of frames in 1 second: {frames_in_1_second}")
    detect()

# Function to stop Detection
def stop_detection():
    global running, cap
    stopButton.pack_forget()
    startButton.place(x=20, y=550)
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

# Fucntion Calculer le taux de fatigue par le nombre de claignement
def tauxFatigueByClaignement(nombreClaignement):
    if nombreClaignement < 10:
        return 10  # Fatigue légère constante
    elif 10 <= nombreClaignement <= 20:
        return 0  # Pas  fatigue
    elif 20 < nombreClaignement <= 30:
        return 15 + (nombreClaignement - 20) * (10 / 10)  # Augmentation linéaire
    elif nombreClaignement > 30:
        return 25  # Fatigue maximale
    else:
        return 0

# Functions calcule taux de fatigue par le nombre de baillement
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
    if len(clins) != 0:
        duree_moyenne=sum(clins)/(len(clins))
    
        if duree_moyenne < 0.3:
            return 0  # Pas de fatigue
        elif 0.3 <= duree_moyenne <= 0.5:
            return 50 * (duree_moyenne - 0.3) / 0.2  # Fatigue croissante
        elif duree_moyenne > 0.5:
            return 50  # Fatigue avancée
        else:
            return 0
    return 0

# Functions calcule taux fatigue avec nombre de headDrop
def tauxFatiguebyHeadDrop(nbre_headDrop):
       if nbre_headDrop == 0:
              return 0, 0  # Pas de fatigue
       elif 1 <= nbre_headDrop <= 3:
              return 1, 20 * nbre_headDrop  # Fatigue modérée
       elif nbre_headDrop > 3:
              return 2, 100  # Fatigue critique
       else:
              return 0, 0


# Function to vizualize all the Face landmarks
def drawFaceLandmarks(landmarks, w, h, frame):
    face_points = []
    # Extract and store all the landmarks in a list
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        face_points.append((x, y))
    
    
    # Draw all the face landmarks and connect them with lines
    for i in range(1, len(face_points)):
        cv2.line(frame, face_points[i - 1], face_points[i], (0, 255, 0), 1)  # Connect landmarks with green line
    # Draw circles at each point to visualize landmarks clearly
    for point in face_points:
        cv2.circle(frame, point, 1, (0, 0, 255), -1)
# Function detect the number of headdrop
def getNumberHeadDrop(landmarks, frame):
    global numberHeadDrop, previous_y
    # Get nose tip landmark (landmark 1 in Mediapipe's face mesh)
    nose_tip = landmarks[1]
    y_position = nose_tip.y
    # Compare current y-position with previous y-position
    if previous_y is not None:
        vertical_movement = abs(previous_y-y_position)
        if vertical_movement > HEAD_DROP_THRESHOLD:
            numberHeadDrop += 1
    previous_y = y_position

    # Problem : the Headdrop should be fast, because of the frame speed.
    return numberHeadDrop

# Function detect the number of Claignement
def getNumberClaignement(ear):
    global blink_count, blinckThreshold, eyes_closed

    if ear < blinckThreshold:
        if not eyes_closed:
            blink_count += 1
            eyes_closed = True
        else:
            eyes_closed = False
    return blink_count

# Funciton detect the number of Clin
def getNumberClin(ear):
    global blinckThreshold, is_closed, blink_start_time, clins
    print("We are inside the getNumberClin")
    print("ear is shold be greather then 0.02 : ", ear)
    print("blinck start time: ", blink_start_time)
    if ear < blinckThreshold:
        print("is closed: ", is_closed)
        if not is_closed:
            # Start timing the blink
            blink_start_time = time.time()
            is_closed = True
    else:
        if is_closed and blink_start_time is not None:
            print("I am inside the else of getNumberClin")
            # Calculate blink duration
            blink_duration = time.time() - blink_start_time
            clins.append(blink_duration)
            is_closed = False
    print("clins: ", clins)
    return clins

# Funcion detect the number of baillement 
def getNumberBaillement(mar):
    global number_baillement, is_yawning
    if mar > yawn_thresh:
        if not is_yawning:
            number_baillement += 1
            is_yawning = True
    else:
        is_yawning = False
    return number_baillement



# Function the Taux of Fatigue global
def getTauxFatigue(nbre_claignement,nbre_baillement,clins,nbre_headDrop):
    
     # Shared variables to store results
    results = {"taux_claignement": None, "taux_baillement": None, "taux_clin": None, "taux_head_drop": None}

    # Worker functions to calculate each fatigue rate
    def calc_claignement():
        results["taux_claignement"] = tauxFatigueByClaignement(nbre_claignement)

    def calc_baillement():
        results["taux_baillement"] = tauxFatigueByBaillement(nbre_baillement)

    def calc_clin():
        results["taux_clin"] = tauxFatigueByClin(clins)

    def calc_head_drop():
        _, results["taux_head_drop"] = tauxFatiguebyHeadDrop(nbre_headDrop)

    # Create threads
    threads = [
        threading.Thread(target=calc_clin),
        threading.Thread(target=calc_head_drop),
        threading.Thread(target=calc_claignement),
        threading.Thread(target=calc_baillement)
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Print the results for checking
    print("Taux Claignement:", results["taux_claignement"])
    print("Taux Baillement:", results["taux_baillement"])
    print("Taux Clin:", results["taux_clin"])
    print("Taux Head Drop:", results["taux_head_drop"])

    # Return the combined fatigue rate
    return results["taux_baillement"]+results["taux_claignement"]+results["taux_clin"] + results["taux_head_drop"] #+

# Function give us the level of Tierd
def getLevelTired(ear, mar, landmarks, frame):
    result = {"nbre_claignement": None, "nbre_baillement": None, "clins": None, "nbre_head_drop": None}
    def calc_getNumberClaignement():
        result['nbre_claignement'] = getNumberClaignement(ear)
    def calc_getNumberBaillement():
        result["nbre_baillement"] = getNumberBaillement(mar)
    def calc_getNumberClin():
        result["clins"] = getNumberClin(ear)
    def calc_getNumberHeadDrop():
        result["nbre_head_drop"] = getNumberHeadDrop(landmarks, frame)
    
    threads = [
        threading.Thread(target=calc_getNumberClaignement),
        threading.Thread(target=calc_getNumberBaillement),
        threading.Thread(target=calc_getNumberClin),
        threading.Thread(target=calc_getNumberHeadDrop)
    ]
    
    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    tauxFatigue = getTauxFatigue(result["nbre_claignement"], result["nbre_baillement"], result["clins"], result["nbre_head_drop"])

    if tauxFatigue <= 20:
        return 0
    elif tauxFatigue > 20 and tauxFatigue <= 40:
        return 1
    elif tauxFatigue > 40 and tauxFatigue <=60:
        return 2
    else:
        return 3

# Function gives alert foreach Level
def alertFatigue(levelTierd, frame):
    global isAlert1, isAlert2, isAlert3
    if levelTierd == 1:
        cv2.putText(frame, "Warning!", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "You are a little bit tired!", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "I suggest getting some rest!", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        isAlert1 = True
    elif levelTierd == 2:
        cv2.putText(frame, "Warning!", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "You are tired!", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Get some rest!", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        isAlert2 = True
    elif levelTierd == 3:
        cv2.putText(frame, "Warning! Warning! Warning!", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Danger!", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "You are too tired!", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Stop! You Must get a rest", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        isAlert3 = True
        threading.Thread(target=play_music, daemon=True).start()
# Function to dispaly the text alert 5s
async def alertBoucle(frame):
    cv2.putText(frame, "Warning! Warning! Warning!", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Danger!", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, "You are too tired!", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Stop! You Must get a rest", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    await asyncio.sleep(5)


# Function to paly the music:
def play_music():
    mixer.music.play()
    time.sleep(5)
    mixer.music.stop()


# Function detect Eyes closed a logtime
def longClosedEye(ear, frame):
    global first_time
    
    duration = 0
    if ear > 0.2:
        first_time = time.time()
    if ear <= 0.2:
        if first_time is None:  # Start tracking if not already started
            first_time = time.time()
        else:  # Calculate duration
            duration = time.time() - first_time

            if duration > 3:  # If eyes are closed for more than 3 seconds
                cv2.putText(frame, "Warning! Warning! Warning!", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Danger!", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "You are too tired!", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Stop! You Must get a rest", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()

# Funciton to initialize all Variable of detections
def inisializeDetect():
    global numberHeadDrop, previous_y, blink_count, eyes_closed, clins, is_closed, is_yawning, number_baillement
    numberHeadDrop = 0 
    previous_y = None
    blink_count = 0
    eyes_closed = False
    clins.clear()
    is_closed = False
    number_baillement = 0
    is_yawning = False
        



# Detection logic
def detect():
    global running
    if not running:
        return

    ret, frame = cap.read()

    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        global flag, yawn_flag, yawn_count, wind, globalTime, frame_count, isAlert1, isAlert2, isAlert3, frameMax
        
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
            
            if isDrawEyesContour:
                drawContour(frame, left_eye)
                drawContour(frame, right_eye)
            if isDrawMouthContour:
                drawContour(frame, mouth)
            if isDrawFaceLandmarks:
                drawFaceLandmarks(landmarks, w, h, frame)
            if isAlert1:
                cv2.putText(frame, "Warning!", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "You are a little bit tired!", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "I suggest getting some rest!", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                frame_count += 1
                if frame_count == frameMax:
                    frame_count = 0
                    isAlert1 = False
            if isAlert2:
                cv2.putText(frame, "Warning!", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "You are tired!", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Get some rest!", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frame_count += 1
                if frame_count == frameMax:
                    frame_count = 0
                    isAlert2 = False
            
            if isAlert3:
                cv2.putText(frame, "Warning! Warning! Warning!", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Danger!", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "You are too tired!", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Stop! You Must get a rest", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frame_count += 1
                if frame_count == frameMax:
                    frame_count = 0
                    isAlert3 = False
            # get level of tired
            levelTired = getLevelTired(ear, mar, landmarks, frame)
            if time.time() - globalTime > globalTimeMax:
               # give an alert if is Fatigue
        
                alertFatigue(levelTired, frame)
                
                globalTime = time.time()
                inisializeDetect()

            
            
            # give an alert of long closed eyes
            #longClosedEye(ear, frame)
            
            

            
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    videoLabel.imgtk = imgtk
    videoLabel.configure(image=imgtk)
    videoLabel.after(10, detect)

# Function to Controlle the Draw of Eyes Contours
def controlleDrawEyesContour():
    global isDrawEyesContour
    isDrawEyesContour = not isDrawEyesContour

# Fonction to controlle the Draw of mouth Contour
def controlleDrawMouthContour():
    global isDrawMouthContour
    isDrawMouthContour = not isDrawMouthContour
# Fonction to controlle the Draw of the face landmarks
def controlleDrawFaceLandmarks():
    global isDrawFaceLandmarks
    isDrawFaceLandmarks = not isDrawFaceLandmarks
# Constantes
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
# Thresholds for detecting head drops
HEAD_DROP_THRESHOLD = 0.08 
numberHeadDrop = 0
previous_y = None 
# Thresholds for detecting Claignements
blink_count = 0
blinckThreshold = 0.15
eyes_closed = False
# Thresholds for detecting Clin
clins = []
blink_start_time = None
is_closed = False
# Threadsholds for detecting Yawning
number_baillement = 0
is_yawning = False
# For starting the openCv
cap = None
running = False
# For long Closed Eyes
first_time = None
# For max 
globalTime = None
globalTimeMax = 30
# Constante for alert
isAlert1 = False
isAlert2 = False
isAlert3 = False
frame_count = 0
frameMax = 70



isDrawEyesContour = False
isDrawMouthContour = False
isDrawFaceLandmarks = False

wind = tk.Tk()
wind.title("OpenEyes")
wind.geometry("640x640")
wind.resizable(False, False)

wind.configure(bg="#121212")  # Dark background

background_image = Image.open("./asset/image-640x640.png")  # Replace with your image file
  # Resize to fixed window size
bg_image = ImageTk.PhotoImage(background_image)

# Create a label to hold the background image
bg_label = tk.Label(wind, image=bg_image)
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
    "width": 10,
    "height": 2
}

label_style = {
    "bg": "#1E88E5",  # Match the background
    "fg": "white",  # White text
    "font": ("Helvetica", 14, "bold")
}

# Video Label
videoLabel = Label(wind, text="Keeping Drivers Awake, One Blink at a Time", **label_style)
videoLabel.pack(pady=20)

# Buttons
startButton = tk.Button(wind, text="Start", command=start_detection, **button_style)
startButton.pack(pady=10)

stopButton = tk.Button(wind, text="Stop", command=stop_detection, **button_style)
stopButton.pack_forget()  # Initially hidden

drawEyesButton = tk.Button(wind, text="Eyes", command=controlleDrawEyesContour, **button_style)
drawEyesButton.pack_forget()

drawMouthButton = tk.Button(wind, text="Mouth", command=controlleDrawMouthContour,**button_style)
drawEyesButton.pack_forget()

drawFaceLandmarksButton = tk.Button(wind, text="Face", command=controlleDrawFaceLandmarks , **button_style)
drawFaceLandmarksButton.pack_forget()
wind.mainloop()
