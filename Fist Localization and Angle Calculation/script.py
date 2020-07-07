import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def make_centroid_point_angle_and_line_img(x, y, frame):
    width, height = frame.shape[1], frame.shape[0]
    img = np.zeros((height, width, 3))

    img[y-10:y+10, x-10:x+10, 0] = 255

    line_slope, line_bias = 0, 0
    line_slope = (height-y)/(x-(width//2)) if (x-(width//2)) != 0 else 9999
    line_bias = - line_slope*(width//2)

    lr = lambda x: x*line_slope + line_bias
    
    for i in range((width//2)-50, (width//2)+50):
        output = height - math.trunc(float(lr(i)))
        if i != width//2 and output > 0 and output > y :
            img[output-5:output+5, i-5:i+5] = 255
    
    
    vect1 = np.array([x-width//2, height-y])
    origin_vect = np.array([1, 0])
    dot = vect1.dot(origin_vect)
    dv1 = np.sqrt((vect1[0]**2) + (vect1[1]**2))
    do = np.sqrt((origin_vect[0]**2) + (origin_vect[1]**2))
    cos_angle = dot/(dv1*do)
    angle = (math.acos(cos_angle)/math.pi)*180
    return img, angle
    
    
fist_cascade = cv2.CascadeClassifier('/Users/mehdi/OneDrive/Desktop/hands2.xml')
c = cv2.VideoCapture(0)

while True:
    ret, frame = c.read()
    fists = fist_cascade.detectMultiScale(frame, 1.1, 7)
    for (x, y, w, h) in fists:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    x, y, w, h = fists[0] if len(fists) >= 1 else (0, 0, 0, 0)     
    img, angle = make_centroid_point_angle_and_line_img(int((x+x+w)/2), int((y+h+y)/2), frame)
    
    if x != 0 or y != 0 or w != 0 or h != 0 :
    
        frame[img != 0] = img[img != 0]   

        cv2.putText(img=frame, text="Angle of the hand to x axis is : "+str(angle), 
                org=(5, 30), 
                fontFace=cv2.FONT_HERSHEY_DUPLEX,  
                fontScale=1, color=(0, 0, 255))
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
c.release()