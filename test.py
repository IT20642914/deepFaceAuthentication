import cv2
import os
import uuid

# Setup paths
VIDEO_PATH = '20240302_234719.mp4'  # Replace with your video file path
POS_PATH = os.path.join('sampeldata', 'positive')
NEG_PATH = os.path.join('sampeldata', 'negative')
ANC_PATH = os.path.join('sampeldata', 'anchor')
IMAGES_PER_FOLDER = 400

# Create the dataset folders if they don't exist
for path in (POS_PATH, NEG_PATH, ANC_PATH):
    os.makedirs(path, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
folder_num = 0
current_folder = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Cut down frame to 250x250px (optional)
    frame = frame[120:120+250, 200:200+250, :]  

    # Create a new folder if we're at the image limit
    if frame_count % IMAGES_PER_FOLDER == 0:
        folder_num += 1
        current_folder = os.path.join(POS_PATH, str(folder_num))  # Alternate between POS_PATH and ANC_PATH
        if folder_num % 2 == 0:
            current_folder = os.path.join(ANC_PATH, str(folder_num))
        os.makedirs(current_folder, exist_ok=True)

    # Save the image in the current folder
    imgname = os.path.join(current_folder, '{}.jpg'.format(uuid.uuid1()))
    cv2.imwrite(imgname, frame)

    # Display the frame with folder info 
    cv2.putText(frame, "Saving to: {}".format(current_folder), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Image Collection', frame)

    frame_count += 1

    # Break if 'q' is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 

