import cv2
import numpy as np

from depthEstimationZoe import DepthEstimatorZoe
from depthEstimation import DepthEstimator
from vehicleDetection import VehicleDetector
from plateDetection import PlateDetector
from generalDetection import GeneralDetector

from utils import *

if __name__ == "__main__":
    # Parameters
    FRAME_RATE = 30
    CONFIDENCE_THRESHOLD = 0.75  # Only show detections with confidence above this threshold
    FOCAL_LENGTH = 800.9  # Focal length in pixels (from camera calibration)
    OFFSET = 0.0  # Offset for further calibration from the camera matrix focal length
    DEVICE = 'cpu'  # 'cpu' or 'cuda:0'
    VIDEO_PATH = 'test_images/vid1.mp4'
    WEBCAM = False

    # Camera intrinsics for custom videos
    camera_matrix = np.array([[660.51081297156020, 0.0, 318.10845757653777],
                              [0.0, 660.51081297156020, 239.95332228230293],
                              [0, 0, 1]])
    dist_coeffs = np.array([0., 0.22202255011309072, 0., 0., -0.50348071005413975])

    # Adjust focal length if using custom camera
    FOCAL_LENGTH = camera_matrix[0][0] + OFFSET

    # Video Capture
    cap = cv2.VideoCapture(0) if WEBCAM else cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    # Depth estimator - K for outdoor, N for indoor, NK for general case (GPU recommended)
    depth_estimator = DepthEstimatorZoe(model_type='K', device=DEVICE)

    # Object detectors
    vehicle_detector = VehicleDetector('./yolov5.pt', device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD)
    plate_detector = PlateDetector(device=DEVICE)
    person_detector = GeneralDetector(device=DEVICE, classes=[0])  # Class 0 is "person"

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        ret, frame = cap.read()
        final_img = frame.copy()

        # Predict depth map
        depth_map = depth_estimator.predict_depth(frame)

        # Detect vehicles and people
        vehicle_boxes = vehicle_detector.detect(frame)
        person_boxes = person_detector.detect(frame)

        if vehicle_boxes is not None:
            # Crop vehicle images and get depth
            vehicle_cropped_images = cropDetection(frame, vehicle_boxes, obj_type='vehicle')
            uncorrected_vehicle_boxes = getMedianDepth(depth_map, vehicle_boxes.copy())
            uncorrected_person_boxes = getMedianDepth(depth_map, person_boxes.copy())

            # Detect plates for each vehicle
            plate_boxes_per_vehicle = [plate_detector.detect(vehicle_img) for vehicle_img, _, _ in vehicle_cropped_images]
            plate_cropped_images_per_vehicle = [
                cropDetection(cropped_img, plate_boxes_per_vehicle[i], obj_type='plate') 
                for i, cropped_img in enumerate(vehicle_cropped_images)
            ]

            # Estimate plate depth and correct depth map
            try:
                known_depths = np.vstack([estimatePlateDepth(plate_cropped_img, final_img, focal_length=FOCAL_LENGTH) 
                                          for plate_cropped_img in plate_cropped_images_per_vehicle])
            except:
                known_depths = None

            corrected_depth_map = depth_estimator.updateDepthEstimates(depth_map, known_depths)
            corrected_vehicle_boxes = getMedianDepth(corrected_depth_map, vehicle_boxes.copy())
            corrected_person_boxes = getMedianDepth(corrected_depth_map, person_boxes.copy())

            # Draw images with and without depth correction
            uncorrected_final_img = drawDetections(final_img, uncorrected_vehicle_boxes, label='vehicle')
            uncorrected_final_img = drawDetections(uncorrected_final_img, uncorrected_person_boxes, label='person')

            corrected_final_img = drawDetections(final_img, corrected_vehicle_boxes, label='vehicle')
            corrected_final_img = drawDetections(corrected_final_img, corrected_person_boxes, label='person')

        # Show corrected and uncorrected images side by side
        img = np.hstack((uncorrected_final_img, corrected_final_img))

        # Initialize writer if not already
        if writer is None:
            writer = cv2.VideoWriter("./results/result.mp4", fourcc, FRAME_RATE, (img.shape[1], img.shape[0]), True)
        
        writer.write(img)
        
        # Display results
        cv2.imshow("Uncorrected(left) vs Corrected(right)", img)

    # Cleanup
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
