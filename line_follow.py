from picamera2 import Picamera2
import time
import cv2
import numpy as np



# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(40, GPIO.OUT)
# GPIO.output(40, GPIO.HIGH)

camera = Picamera2()

camera.resolution = (640, 360)
camera.rotation = 180
modes = camera.sensor_modes
mode = modes[1]
print('mode selected: ', mode)
camera_config = camera.create_still_configuration(raw={'format': mode['unpacked']}, sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']}, main={"size":(640, 480)})
camera.configure(camera_config)
camera.resolution = (640, 360)
camera.rotation = 180
print("cam initialized")
camera.start()

time.sleep(0.1)

while True:	
    image = camera.capture_array()
    Blackline = cv2.inRange(image, (180,180,180), (255,255,255))
    cv2.imshow("White",Blackline)
    kernel = np.ones((3,3), np.uint8)
    Blackline = cv2.erode(Blackline, kernel, iterations=5)
    Blackline = cv2.dilate(Blackline, kernel, iterations=9)	
    contours_blk, hierarchy_blk = cv2.findContours(Blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
    if len(contours_blk) > 0:
        blackbox = cv2.minAreaRect(contours_blk[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox
        if ang < -45 :
            ang = 90 + ang
        if w_min < h_min and ang > 0:	  
            ang = (90-ang)*-1
        if w_min > h_min and ang < 0:
            ang = 90 + ang	  
        setpoint = 320
        error = int(x_min - setpoint) 
        ang = int(ang)	 
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),3)	 
        cv2.putText(image,str(ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image,str(error),(10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(image, (int(x_min),200 ), (int(x_min),250 ), (255,0,0),3)
	 
	  	
    cv2.imshow("orginal with line", image)	
    key = cv2.waitKey(1) & 0xFF	
    if key == ord("q"):
        break
