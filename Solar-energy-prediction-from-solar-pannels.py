import cv2
import numpy as np
print("Packages Imported sucessfully")
img1 = cv2.imread(r"C:\Users\11373\Documents\cse\DESIGN PROJECT\images\img1.png")
img2 = cv2.imread(r"C:\Users\11373\Documents\cse\DESIGN PROJECT\images\img2.png")
img3 = cv2.imread(r"C:\Users\11373\Documents\cse\DESIGN PROJECT\images\img3.png")
img4 = cv2.imread(r"C:\Users\11373\Documents\cse\DESIGN PROJECT\images\img4.png")
print("Pre-images were loaded sucessfully. Converting into Grey-scale images....")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
print("Sucessfully converted into grey scale images.")
#Create an instance of Oriented fast and Rotated BRIEF feature detector
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
kp3, des3 = orb.detectAndCompute(gray3, None)
kp4, des4 = orb.detectAndCompute(gray4, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    matches1 = bf.match(des1, des)
    matches2 = bf.match(des2, des)
    matches3 = bf.match(des3, des)
    matches4 = bf.match(des4, des)
    img = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))
    if len(matches1) > 125 or len(matches2) > 125 or len(matches3) > 125 or len(matches4) > 125:
        print(len(matches1),len(matches2),len(matches3),len(matches4))
        
    matches_lengths = []
    for matches in [matches1, matches2, matches3, matches4]:
        length = len(matches) - 125
        if length > 0:
            matches_lengths.append(length)
    if len(matches_lengths) > 0:
        print("Matches exceeding threshold: ", matches_lengths)
        avg_matches = np.mean(matches_lengths)
        print("Average matches exceeding threshold: ", avg_matches)
        print("The average energy can be genearated by the solar pannel is: ",avg_matches*3.25,"KW/h")
    cv2.imshow('frame', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
