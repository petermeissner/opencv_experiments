import cv2 as cv

def face_detect(img):
    frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        img = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            img = cv.circle(img, eye_center, radius, (255, 0, 0 ), 4)

    # return 
    return img

    
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
face_cascade.load('data/haarcascades/haarcascade_frontalface_alt.xml')
eyes_cascade.load('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')


# -- Execute detection and write to file
fname     = "peter_tshirt.jpg"

img       = cv.imread(fname)
img_faces = face_detect(img)

cv.imwrite(
    filename = (fname + "_face_marked.png"), 
    img      = img_faces
  )

