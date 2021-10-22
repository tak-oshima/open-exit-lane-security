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
    args = parser.parse_args()

    # Assign algorithms
    backSub = selectBGS(args.detector)
    tracker = selectSOT(args.tracker)
    mot_tracker = Sort(max_age=10000, min_hits=3, iou_threshold=0.001)
    trajectories = {}

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
        if frame is None:
            break

        frame = cv.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv.INTER_AREA)
        fgMask = backSub.apply(frame)
        if args.detector == 'GMG':
            fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

        mask = refineFGMask(fgMask)
        contours= cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv.boundingRect(cnt)
                detections.append([x, y, x + w, y + h, 1])

                # Countour Approximation
                # epsilon = 0.1 * cv.arcLength(cnt, True)
                # approx = cv.approxPolyDP(cnt, epsilon, True)

                # Convex Hull
                # hull = cv.convexHull(cnt)

                # Draw raw contours
                # cv.drawContours(frame, [hull], -1, (0, 255, 0), 1)

                # Moments
                # moments = cv.moments(cnt)
                # cx = int(moments['m10']/moments['m00'])
                # cy = int(moments['m01']/moments['m00'])

        # Pre-process detections and update SORT tracker
        detections = np.array(detections) if detections else np.array(np.empty((0, 5)))
        assignments = mot_tracker.update(detections)

        for x1, y1, x2, y2, id in assignments:
            cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
            if id in trajectories.keys():
                trajectories[id].append((cx, cy))
            else:
                trajectories[id] = [(cx, cy)]

            # Draw bounding boxes and IDs
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BB_COLORS[int(id % 13)], 2)
            cv.putText(frame, str(int(id)), (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, BB_COLORS[int(id % 13)], 2)
            cv.circle(frame, (cx, cy), 1, BB_COLORS[int(id % 13)], 2)

            # Draw trajectories
            prev_point = trajectories[id][0]
            for point in trajectories[id]:
                cv.line(frame, prev_point, point, thickness=1, color=BB_COLORS[int(id % 13)], shift=0, lineType=cv.LINE_AA)
                prev_point = point
        
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.45 , (0, 0, 0))
        

        cv.imshow('Frame', frame)
        cv.moveWindow('Frame', 0, int(height * scale + 28))
        cv.imshow('FG Mask', mask)
        
        keyboard = cv.waitKey(15)
        if keyboard == 'q' or keyboard == 27:
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()