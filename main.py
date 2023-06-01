import cv2
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
import torch

from depthEstimationZoe import DepthEstimatorZoe
from depthEstimation import DepthEstimator
from vehicleDetection import VehicleDetector
from plateDetection import PlateDetector
from generalDetection import GeneralDetector

'''
These extra functions should be moved to a utils file later
'''
def drawDetections(img, detections, label='vehicle'):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Draw detections and add depth values as a label to the rectangle
    color = (0, 255, 0) if label == 'vehicle' else (0, 255, 255) # green for vehicles, yellow otherwise
    detection_img = img.copy()
    for index, row in detections.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(detection_img, (xmin, ymin), (xmax, ymax), color, 2)
        depth_label = "Depth: {:.2f} m".format(round(row['depth'], 2))
        # Add black background panel to depth label
        label_size, _ = cv2.getTextSize(depth_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        panel_size = (label_size[0]+10, label_size[1]+10)
        panel_pos = (xmin, ymin - panel_size[1])
        cv2.rectangle(detection_img, panel_pos, (panel_pos[0]+panel_size[0], panel_pos[1]+panel_size[1]), (0,0,0), -1)
        cv2.putText(detection_img, depth_label, (panel_pos[0]+5, panel_pos[1]+label_size[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return detection_img

# Gets the minimum depth from depth map within a bounding box, currently used to determine the depth for detected vehicles.
def getMinDepth(depth, detections):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Get depth of each detection
    # Add depth to detection data frame as a new column
    depth_per_detection = []
    for i, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        depth_values = depth[ymin:ymax, xmin:xmax].flatten()
        depth_values = depth_values[depth_values != 0] # remove zeros
        depth_per_detection.append(depth_values.min())

    # Add depth to detection data frame as a new column
    detections["depth"] = depth_per_detection

    return detections

# Taking the median is probably better, that way if any object gets infront of the detected car, it is less likely to pick the depth value of that object over the car.
def getMedianDepth(depth, detections):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Get depth of each detection
    # Add depth to detection data frame as a new column
    depth_per_detection = []
    for i, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        depth_values = depth[ymin:ymax, xmin:xmax].flatten()
        depth_values = depth_values[depth_values != 0] # remove zeros
        depth_per_detection.append(np.median(depth_values))

    # Add depth to detection data frame as a new column
    detections["depth"] = depth_per_detection

    return detections

def estimatePlateDepth(detections, org_img, real_world_dims=(0.520, 0.110), focal_length=800): # Default plate size for korea 335mm width, 170mm height
    '''
    Takes cropped images of detected plates and estimates the depth from known world size of plate.
    All of the pixels belonging to the detections are then assigned to this estimated depth
    The function returns an array of tuples (x, y, z) with x, y coordinates of plate pixel in image and z estimated depth.
    detections[i][0] is the image, detections[i][1] is the xmin, ymin, xmax, ymax coordinates in the original image for plate i
    '''
    pixel_coords = np.empty((0, 4))
    for i, detection in enumerate(detections):
        img, location, confidence = detection
        xmin, ymin, xmax, ymax = location
        # Calculate depth from known plate size
        plate_width, plate_height = real_world_dims
        plate_width_pixels = xmax - xmin
        plate_height_pixels = ymax - ymin
        # Calculate depth in meters from known plate size 
        depth_width = (focal_length*plate_width) / plate_width_pixels
        depth_height = (focal_length*plate_height) / plate_height_pixels
        depth = (depth_width + depth_height) / 2
        # Create a 2D array of depth values
        depth_array = np.full((plate_height_pixels, plate_width_pixels), depth)
        confidence_array = np.full((plate_height_pixels, plate_width_pixels), confidence)
        # Create arrays of x and y coordinates
        x_coords = np.arange(xmin, xmax).astype(np.int32)
        y_coords = np.arange(ymin, ymax).astype(np.int32)
        # Create a grid of x and y coordinates
        xx, yy = np.meshgrid(x_coords, y_coords)
        # Stack the x, y, and depth arrays
        pixel_coords_d = np.dstack((yy,xx, depth_array, confidence_array))
        # Reshape the pixel_coords array into a 2D array
        pixel_coords_d = pixel_coords_d.reshape(-1, 4)
        # Append the pixel_coords to the estimates list
        pixel_coords = np.append(pixel_coords, pixel_coords_d, axis=0)

        # Visualize for testing
        # Draw bounding box on original image and add depth label
        cv2.rectangle(org_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        depth_label = "Depth: {:.2f} m".format(round(depth, 2))
        # Add black background panel to depth label
        label_size, _ = cv2.getTextSize(depth_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        panel_size = (label_size[0]+10, label_size[1]+10)
        panel_pos = (xmin, ymin - panel_size[1])
        cv2.rectangle(org_img, panel_pos, (panel_pos[0]+panel_size[0], panel_pos[1]+panel_size[1]), (0,0,0), -1)
        cv2.putText(org_img, depth_label, (panel_pos[0]+5, panel_pos[1]+label_size[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return pixel_coords

def cropDetection(src_img, detections, obj_type='vehicle'):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Draw detections and add depth values as a label to the rectangle
    # Returns a list of tuples containing the cropped images and location of the cropped image in the original image
    if obj_type == 'vehicle':
        detection_img = src_img.copy()
        img = src_img
    else:
        detection_img = src_img[0].copy()
        img = src_img[0]
        
    cropped_images = []
    for i, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
        confidence = detection['confidence']
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        label = f"cropped_{i}"
        cv2.rectangle(detection_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(detection_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cropped_img = img[ymin:ymax, xmin:xmax]
        location = (xmin, ymin, xmax, ymax)
        if obj_type == 'plate':
            org_xmin, org_ymin, org_xmax, org_ymax = src_img[1]
            location = (org_xmin + xmin, org_ymin + ymin, org_xmin + xmax, org_ymin + ymax)

        cropped_images.append((cropped_img, location, confidence))
    return cropped_images

def showCroppedDetection(cropped_images, label=''):
    # Uses cv2 to show all the detections in a grid on a single window
    num_cols = 4
    num_rows = -(-len(cropped_images) // num_cols)  # Ceiling division to calculate number of rows
    grid_size = 100
    padding = 10
    output_img = np.zeros((num_rows * (grid_size + padding) + padding, num_cols * (grid_size + padding) + padding, 3), dtype=np.uint8)
    output_img.fill(255)
    for i, cropped_img in enumerate(cropped_images):
        row = i // num_cols
        col = i % num_cols
        x = col * (grid_size + padding) + padding
        y = row * (grid_size + padding) + padding
        resized_img = cv2.resize(cropped_img[0], (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        output_img[y:y+grid_size, x:x+grid_size] = resized_img
    cv2.imshow(f"Cropped Detections {label}", output_img)

def getEdges(img):
    # Convert image to grayscale
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)
    # Find the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)
    # Draw the rectangle on the original image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return img

if __name__ == "__main__":
    # Parameters
    FRAME_RATE = 30 
    CONFIDENCE_THRESHOLD = 0.75 # Only show detections with confidence above this threshold
    FOCAL_LENGTH = 800.9 # Focal length is supposed to be in pixels. This is what we get from the camera calibration.
    OFFSET = 0.0 # Offset to the focal length value for further calibration from camera matrix focal length
    DEVICE = 'cpu' # cpu or cuda:0
    VIDEO_PATH = 'test_images/recorded1_undistorted.mp4'
    WEBCAM = False

    # Camera intrinsics for when we use our own videos
    camera_matrix = np.array([[6.6051081297156020e+02, 0.0, 3.1810845757653777e+02], 
                              [0.0, 6.6051081297156020e+02, 2.3995332228230293e+02],
                              [0, 0, 1]])
    dist_coeffs = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])

    # Change the focal length if using own camera
    FOCAL_LENGTH = camera_matrix[0][0] + OFFSET
    
    cap = cv2.VideoCapture(0) if WEBCAM else cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    # Use MiDaS_small if running on CPU, otherwise DPT_Large or ZoeDepth's K variant model.
    depth_estimator = DepthEstimatorZoe(model_type='K', device=DEVICE) # K for outdoors scene, N for indoors, NK for general case. Don't even bother with this one without GPU. Does produce better depth maps though
    #depth_estimator = DepthEstimator(model_type='MiDaS_small', device=DEVICE) # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.

    vehicle_detector = VehicleDetector('./yolov5.pt', device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD)
    plate_detector = PlateDetector(device=DEVICE)

    person_detector = GeneralDetector(device=DEVICE, classes=[0]) # 0 class is only classifying "person"
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        ret, frame = cap.read()
        final_img = frame.copy()
        depth_map = depth_estimator.predict_depth(frame)

        # Detect vehicles
        vehicle_boxes = vehicle_detector.detect(frame)
        # Detecting people
        person_boxes = person_detector.detect(frame)

        if vehicle_boxes is not None:
            vehicle_cropped_images = cropDetection(frame, vehicle_boxes, obj_type='vehicle')
            uncorrected_vehicle_boxes = getMedianDepth(depth_map, vehicle_boxes.copy())
            uncorrected_person_boxes = getMedianDepth(depth_map, person_boxes.copy()) 
            plate_boxes_per_vehicle = [plate_detector.detect(vehicle_img) for vehicle_img, _, _ in vehicle_cropped_images]
            plate_cropped_images_per_vehicle = [cropDetection(cropped_img, plate_boxes_per_vehicle[i], obj_type='plate') for i, cropped_img in enumerate(vehicle_cropped_images)]

            # Estimate depth of plate and correct depth map
            try:
                known_depths = np.vstack([estimatePlateDepth(plate_cropped_img, final_img, focal_length=FOCAL_LENGTH) for plate_cropped_img in plate_cropped_images_per_vehicle])
            except:
                known_depths = None

            corrected_depth_map = depth_estimator.updateDepthEstimates(depth_map, known_depths)
            corrected_vehicle_boxes = getMedianDepth(corrected_depth_map, vehicle_boxes.copy())
            corrected_person_boxes = getMedianDepth(corrected_depth_map, person_boxes.copy())

            # Draw the images with and without depth correction
            uncorrected_final_img = drawDetections(final_img, uncorrected_vehicle_boxes, label='vehicle')
            uncorrected_final_img = drawDetections(uncorrected_final_img, uncorrected_person_boxes, label='person')
            corrected_final_img = drawDetections(final_img, corrected_vehicle_boxes, label='vehicle')
            corrected_final_img = drawDetections(corrected_final_img, corrected_person_boxes, label='person')

        # Show corrected and uncorrected images side by side
        img = np.hstack((uncorrected_final_img, corrected_final_img))
        #img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        #img = corrected_final_img
        cv2.imshow("Uncorrected(left) vs Corrected(right)", img)

        if writer is None:
            writer = cv2.VideoWriter("./results/result.mp4", fourcc, FRAME_RATE, (img.shape[1], img.shape[0]), True)
        writer.write(img)
        #cv2.imshow("Results without depth map correction", uncorrected_final_img)
        #cv2.imshow('Results using corrected depth map', corrected_final_img)

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
