from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
from pygame import mixer
from tkinter import *
from tkinter import PhotoImage,Canvas,font
from PIL import ImageTk , Image
mixer.init()
mixer.music.load("beep-09.mp3")

def eyeAspectRatio(eye):
    X=dist.euclidean(eye[1],eye[5])
    Y=dist.euclidean(eye[2],eye[4])
    Z=dist.euclidean(eye[0],eye[3])
    ear=(X+Y)/(2.0 *Z)
    return ear
def sleepiness():
    count=0
    earThresh=0.3
    earFrames=48
    shapePredictor="shape_predictor_68_face_landmarks.dat"
    cam=cv2.VideoCapture(0)
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(shapePredictor)

    (lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        _,frame=cam.read()
        frame=imutils.resize(frame,width=800,height=500)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rects=detector(gray,0)


        for rect in rects:
            shape=predictor(gray,rect)
            shape=face_utils.shape_to_np(shape)

            leftEye=shape[lStart:lEnd]
            rightEye=shape[rStart:rEnd]
            leftEAR=eyeAspectRatio(leftEye)
            rightEAR=eyeAspectRatio(rightEye)
            ear=(leftEAR+rightEAR)/2.0
        
            leftEyeHull=cv2.convexHull(leftEye)
            rightEyeHull=cv2.convexHull(rightEye)
            cv2.drawContours(frame,  [leftEyeHull],-1,(0,0,255),1)
            cv2.drawContours(frame,  [rightEyeHull],-1,(0,0,255),1)

            if ear<earThresh:
                count+=1
                if count>=earFrames:
                    cv2.putText(frame,"ALERT...SLEEPINESS DETECTED",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    mixer.music.play()
            else:
                count=0
        cv2.imshow("Frame",frame)
        key=cv2.waitKey(1) & 0xFF

        if key==ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

window=Tk()
window.geometry("900x700")
window.title("SLEEPINESS Detection")
c=Canvas(window,height=400,width=400)


heading_font = font.Font(family="Arial", size=35, weight="bold")
heading_label = Label(window, text="SLEEPINESS DETECTION", font=heading_font)
heading_label.pack()
c.pack()
my_pic=PhotoImage(file="pic.png")
c.create_image(0,0, anchor="nw", image=my_pic)
my_label=Label(window,image=my_pic)
button=Button(c,text="Start camera ",command=sleepiness,font=("Conic sans",20),fg="white",
              bg="red", activeforeground="#00FF00",
             activebackground="black",anchor="e")

my_label.pack(pady=20)
button.place(relx=0.5,rely=0.5,anchor="e")
button.pack()
window.mainloop()
