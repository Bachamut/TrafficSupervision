import cv2


class ContourDetector:

    def __init__(self, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, contour_min_area=400):
        self.mode = mode
        self.method = method
        self.contour_min_area = contour_min_area

    def detect_contours(self, background_thresh, roi):

        contours, hierarchy = cv2.findContours(background_thresh, self.mode, self.method)
        detections = list()

        for cnt in contours:
            # Calculate area and remove small objects
            area = cv2.contourArea(cnt)
            if area > self.contour_min_area:
                cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detections.append([x, y, w, h])

        return detections
