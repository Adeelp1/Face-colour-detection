import cv2
 
#Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize variables
tracker = None
tracking = False

# Define color ranges and labels
COLOR_RANGES = {
    "YELLOW": {
        "lower": (25,100,100),
        "upper": (35,255,255),
        "rectangle_color": (0, 255, 255)
    },
    "RED": {
        "lower": (0, 50, 50),
        "upper": (10, 255, 255),
        "rectangle_color": (0, 0, 255)
    },
    "GREEN": {
        "lower": (40, 20, 50),
        "upper": (90, 255, 255),
        "rectangle_color": (0, 255, 0)
    },
    "BLUE": {
        "lower": (100, 50, 50),
        "upper": (130, 255, 255),
        "rectangle_color": ((255, 0, 0))
    }
} 

CONTOUR_AREA_THRESHOLD = 1000  # Minimum area to consider a contour

# Initialize video capture
cap = cv2.VideoCapture(0)
 
while True:
    read_ok, img = cap.read()
    if not read_ok:
        print("Failed to read from camera.")
        break
  
    img = cv2.resize(img, (640, 480))
    img = cv2.flip(img,1)
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for color_name, color_info in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, color_info["lower"], color_info["upper"])
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop through the contours and draw a rectangle around them
        for cnt in contours:
            if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x,y), (x + w, y + h), color_info["rectangle_color"], 2)
                cv2.putText(img, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_info["rectangle_color"], 2)
    
    # Face detection
    # If not tracking, detect face O(W * H * log(W * H))
    if not tracking:
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            tracker = cv2.TrackerKCF_create() # Initialize the tracker
            tracker.init(img, (x, y, w, h))
            tracking = True
    # If tracking, update the tracker O(1)
    if tracking:
        success, bbox = tracker.update(img)
        if success:
            # draw bounding box around the tracked face
            (x, y, w, h) = map(int, bbox)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 4)
            cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        else:
            # Tracking failed, switch back to detection mode
            tracking = False 
  
    cv2.imshow('Color and Face Detection Output', img)
     
    # Close video window by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()