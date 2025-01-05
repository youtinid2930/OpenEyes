import tkinter as tk
import cv2
from PIL import Image, ImageTk
wind = tk.Tk()
wind.title("vision")
#initialisation la capture video
capture = cv2.VideoCapture(0)
#create label
video_label = tk.Label(wind)
video_label.pack()
#la transformation actuel
transformation_actuel = None
# Charger le classificateur de visage
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
#fonction pour afficher le video
def afficher_video():
    global video_label, transformation_actuel
    # Lire une image de la webcam
    ret, frame = capture.read()
    if ret:
        #appliquer la transformation selectionnee:
        if transformation_actuel == "grayscale":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif transformation_actuel == "blur":
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif transformation_actuel == "canny":
            frame = cv2.Canny(frame, 50, 150)
        elif transformation_actuel == "face":
            #conversion en niveau de gris pour la detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #detections des visages
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Dessiner des rectangles autour des visages detecter
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # on a pas besion de convetir BGR a RGB si la transformation est en gris and canny
        if transformation_actuel in ["grayscale", "canny"]:
            # Covertir en image PIL
            image = Image.fromarray(frame)
        else:
            # Convertir de BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Covertir en image PIL
            image = Image.fromarray(frame)
        # Convertir en image Tkinter
        image_tk = ImageTk.PhotoImage(image=image)
        # Mettre a jour la label
        video_label.imgtk = image_tk
        video_label.configure(image=image_tk)
    # Appeler la fonction apres 10 ms
    video_label.after(10, afficher_video)

#fonction pour faire la transformation
def setTransformation(transformation):
    global transformation_actuel
    transformation_actuel = transformation
#add buttons:
#pour le niveau de gris
niveauGrisButton = tk.Button(wind, text="niveaux de gris", command=lambda: setTransformation("grayscale"))
niveauGrisButton.pack(side=tk.LEFT, padx=5, pady=5)
#pour flou gaussien
flouGaussienButton = tk.Button(wind, text="Flou Gaussien", command=lambda: setTransformation("blur"))
flouGaussienButton.pack(side=tk.LEFT, padx=5, pady=5)
#pour Contours (Canny)
contoursButton = tk.Button(wind, text="Contours (Canny)", command=lambda: setTransformation("canny"))
contoursButton.pack(side=tk.LEFT, padx=5, pady=5)
#pour la reinitialisation:
resetButton = tk.Button(wind, text="Réinitialiser", command=lambda: setTransformation(None))
resetButton.pack(side=tk.LEFT, padx=5, pady=5)
#pour la detections de visage:
faceButton = tk.Button(wind, text="Détection de Visage", command=lambda: setTransformation("face"))
faceButton.pack(side=tk.LEFT, padx=5, pady=5)


afficher_video()
wind.mainloop()
#liberer la capture video
capture.release()
