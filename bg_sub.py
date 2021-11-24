from __future__ import print_function
from sort import *
import cv2 as cv
import argparse
import numpy as np
import sys
import imutils

BB_COLORS = [
    (214, 30, 30),
    (214, 150, 30),
    (237, 230, 24),
    (163, 227, 52),
    (99, 214, 11),
    (26, 235, 162),
    (15, 242, 231),
    (5, 187, 242),
    (0, 114, 245),
    (111, 78, 230),
    (150, 70, 235),
    (208, 45, 237),
    (240, 29, 198)
]

BREACH_HISTORY = 10
BREACH_SENSITIVITY = 0.5

def selectBGS(name):
    if name == 'MOG':
        subtr = cv.bgsegm.createBackgroundSubtractorMOG()
    elif name == 'MOG2':
        subtr = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
    elif name == 'GMG':
        subtr = cv.bgsegm.createBackgroundSubtractorGMG()
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    elif name == 'KNN':
        subtr = cv.createBackgroundSubtractorKNN()
    else:
        sys.exit('No detector named {}'.format(name))
    return subtr

def selectSOT(name):
    if name == 'BOOSTING':
        tracker = cv.TrackerBoosting_create()
    elif name == 'MIL':
        tracker = cv.TrackerMIL_create()
    elif name == 'KCF':
        tracker = cv.TrackerKCF_create()
    elif name == 'TLD':
        tracker = cv.TrackerTLD_create()
    elif name == 'MEDIANFLOW':
        tracker = cv.TrackerMedianFlow_create()
    elif name == 'GOTURN':
        tracker = cv.TrackerGOTURN_create()
    elif name == 'MOSSE':
        tracker = cv.TrackerMOSSE_create()
    elif name == 'CSRT':
        tracker = cv.TrackerCSRT_create()
    else:
        sys.exit('No tracker named {}'.format(name))
    return tracker

def refineFGMask(fgMask):
    # mask = cv.GaussianBlur(fgMask, (21, 21), 0)
    _, mask = cv.threshold(fgMask, 244, 255, cv.THRESH_BINARY)
    # mask = cv.erode(mask, None, iterations=3)
    # mask = cv.dilate(mask, None, iterations=3)
    return mask


def main():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='camera')
    parser.add_argument('--detector', type=str, help='Background subtraction method (MOG, MOG2, GMG, KNN).', default='MOG2')
    parser.add_argument('--tracker', type=str, help='Tracking metod (BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT).', default='CSRT')
    parser.add_argument('--size', type=int, help='Minumum size of object to be detected (pixel^2).', default=300)
    parser.add_argument('--breach-dir', type=str, help='Direction that triggers a security breach (left, right).', default='right')
    args = parser.parse_args()

    # Assign algorithms
    backSub = selectBGS(args.detector)
    tracker = selectSOT(args.tracker)
    mot_tracker = Sort(max_age=10000, min_hits=3, iou_threshold=0.001)
    trajectories = {}
    breach_dir = 1 if args.breach_dir == 'right' else -1

    # Read video from camera or file
    if args.input == 'camera':
        capture = cv.VideoCapture(0)
    else:
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    # Display specifications to console
    width, height, scale  = capture.get(3), capture.get(4), 0.8 
    print("detector: {}".format(args.detector))
    print("resolution: {} * {}".format(width * scale, height * scale))

    # Initiate video processing
    while True:
        detections = []
        ret, frame = capture.read()

        # Skip over broken frame
        if frame is None:
            continue

        # frame = cv.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv.INTER_AREA)
        fgMask = backSub.apply(frame)
        if args.detector == 'GMG':
            fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

        mask = refineFGMask(fgMask)
        contours= cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > args.size:
                x, y, w, h = cv.boundingRect(cnt)
                detections.append([x, y, x + w, y + h, 1])

        # Pre-process detections and update SORT tracker
        detections = np.array(detections) if detections else np.array(np.empty((0, 5)))
        assignments = mot_tracker.update(detections)

        for x1, y1, x2, y2, id in assignments:
            cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
            if id in trajectories.keys():
                trajectories[id].append((cx, cy))
            else:
                trajectories[id] = []   # not recording initial position because of buggy behavior of SORT

            # Calculate breach confidence
            breach_confidence = 0
            if len(trajectories[id]) > BREACH_HISTORY:
                for i in range(len(trajectories[id]) - BREACH_HISTORY, len(trajectories[id])):
                    if trajectories[id][i][0] * breach_dir > trajectories[id][i - 1][0] * breach_dir:
                        breach_confidence += 1

            # color = (0, 0, 255) if breach_confidence > 7 else BB_COLORS[int(id % 13)]
            color = (0, 0, 255) if breach_confidence > BREACH_HISTORY * BREACH_SENSITIVITY else (0, 255, 0)
            text = 'ALERT' if breach_confidence > BREACH_HISTORY * BREACH_SENSITIVITY else ''

            # Draw bounding box and ID
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv.putText(frame, text, (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Draw trajectory
            for i in range(len(trajectories[id])):
                if i == 50:
                    break
                idx = len(trajectories[id]) - i - 1
                cv.circle(frame, center=trajectories[id][idx], radius=int(10 * (1 - i / 50)), color=color, thickness=-1, lineType=cv.LINE_AA)

        
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.45 , (0, 0, 0))
        
        cv.imshow('FG Mask', mask)
        cv.imshow('Frame', frame)
        # cv.moveWindow('FG Mask', 0, int(height * scale + 28))
        
        
        keyboard = cv.waitKey(15)
        if keyboard == 'q' or keyboard == 27:
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()