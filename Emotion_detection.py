import f_emotion_detection as fed
import cv2
from imutils.video import VideoStream
import imutils
import argparse
import time

parser = argparse.ArgumentParser(description="Emotions detection")
parser.add_argument('--input', type=str, default= 'webcam',
                    help="webcam or image")
parser.add_argument('--path', type=str,
                    help="path of image")
args = vars(parser.parse_args())

#instantiate detector
type_input = args['input']
Detector = fed.predict_emotions()

if type_input == 'webcam':
    # ----------------------------- video -----------------------------
    #take input data
    vs = VideoStream(src=0).start()
    while True:
        star_time = time.time()
        im = vs.read()
        im = cv2.flip(im, 1)
        im = imutils.resize(im, width=720)
        #detect face
        emotions,boxes_face = Detector.get_emotion(im)
        #display
        if len(emotions)!=0:
            img_post = fed.bounding_box(im,boxes_face,emotions)
        else:
            img_post = im 

        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(im,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('emotion_detection',img_post)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break