import cv2
import numpy as np

import frame_content
from background_subtractor import BackgroundSubtractor
from box import Box
from contour_detector import ContourDetector
from frame_preparation import FramePreparation
from dnn_detector import DnnDetector
from entity import Entity
from id_handler import IdHandler
from entity_handler import EntityHandler


class MediaReader:

    def __init__(self):
        media_dir = "media/video/los_angeles.mp4"
        # media_dir = "media/video/highway.mp4"
        self.media = cv2.VideoCapture(media_dir)
        self.frame_number = 1

        self.dnn_detector = DnnDetector()
        self.contour_detector = ContourDetector()
        self.entity_handler = EntityHandler()
        self.id_handler = IdHandler()
        self.frame_preparation = FramePreparation()
        self.background_subtractor = BackgroundSubtractor()

        self.background = self.background_subtractor.subtract_background()

        self.height = self.media.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.media.get(cv2.CAP_PROP_FRAME_WIDTH)

    def main_loop(self):

        while True:
            ret, frame = self.media.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), None, .5, .5)
            clear_frame = frame.copy()
            dnn_frame = frame.copy()

            roi = frame.copy()[0:int(self.height), 0:int(self.width)]

            frame = self.frame_preparation.convert_grayscale(frame)
            # frame = self.frame_preparation.apply_blur(frame)

            background_mask = self.background.apply(frame)
            _, background_thresh = self.frame_preparation.apply_threshold(background_mask)

            detections = self.contour_detector.detect_contours(background_thresh, roi)

            detection_rectangles = list()
            rectangles_mask = np.zeros(frame.shape[:2], dtype='uint8')
            for rect in detections:
                x, y, w, h = rect
                cut_detection = frame[y:y + h, x:x + w]
                text = f'{x, y, w, h}'
                print(f'x:{x}, y:{y}, w:{w}, h:{h}')
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                cv2.putText(roi, text, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)

                blank = np.zeros(frame.shape[:2], dtype='uint8')
                rectangle = cv2.rectangle(blank.copy(), (x, y), (x + w, y + h), 255, -1)
                # cv2.imshow("Detection Rectangle", rectangle)
                detection_rectangles.append(rectangle)
                rectangles_mask = cv2.add(rectangles_mask, rectangle)

            # cv2.imshow('Rectangles Mask', rectangles_mask)
            #     cut_detection = cv2.bitwise_and(clear_frame, clear_frame, mask=rectangle)
            #     cut_detection = cut_detection[y:y+h, x:x+w]

                cv2.imshow('Cut Detection', cut_detection)
            masked = cv2.bitwise_and(clear_frame, clear_frame, mask=rectangles_mask)
            cv2.imshow('Masked Frame', masked)

            run_dnn = True
            if run_dnn:
                class_ids, scores, boxes = self.dnn_detector.detect(masked)

                for index, class_id in enumerate(class_ids):
                    box_dimension = boxes[index]
                    score = scores[index]
                    box = Box(box_dimension)
                    box_cx, box_cy = box.get_center_coordinates()

                    entity = self.entity_handler.find_convergent_entity(box)

                    if entity is None or self.frame_number == 1:
                        obj_id = self.id_handler.create_next_id(self.frame_number)
                        entity = Entity(box, class_id, obj_id)
                        # entity.attach_box(box)
                        entity.box.color = (0, 255, 0)
                        entity.box.label = box.create_label()

                        entity.center_position = (box_cx, box_cy)
                        entity.create_position_history(self.frame_number)
                        self.entity_handler.tracked_entity[entity] = []

                    else:
                        entity.box.color = (255, 0, 0)
                        entity.box.update_box_position(box)
                        entity.update_entity_center_position(box_cx, box_cy)
                        entity.position_history.update_position_history(self.frame_number)

                        entity.entity_id_history.append(entity.entity_id.number)

                    entity.box.display_border(dnn_frame)
                    entity.box.label.display_border(dnn_frame)
                    entity.box.label.display_title(dnn_frame)
                    entity.display_id(dnn_frame)
                    entity.update_confidence_history(self.frame_number, class_id, score)

                    print(f'Frame No.{self.frame_number}|Id:{entity.entity_id.number} ffd({entity.entity_id.date_of_creation}) - {entity.confidence_history[self.frame_number]}')

                if self.frame_number > 1:
                    for entity in self.entity_handler.tracked_entity:
                        EntityHandler.update_previous_position(entity)

                if self.frame_number > 2:
                    for entity in self.entity_handler.tracked_entity:
                        entity.predicted_position = EntityHandler.predict_entity_next_position(entity)
                        cv2.circle(frame, entity.center_position, 2, (0, 255, 0), 1)
                        if entity.predicted_position is not None:
                            cv2.circle(frame, entity.predicted_position, 2, (0, 0, 255), 1)

            frame_content.display_frame_number(frame, self.frame_number)
            # frame_content.display_frame_number(blur, self.frame_number)
            # cv2.imshow("Frame", frame)
            cv2.imshow("Background Thresh", background_thresh)
            cv2.imshow("Background Mask", background_mask)
            # cv2.imshow('Blur', blur)
            # cv2.imshow('Clear Frame', clear_frame)
            cv2.imshow('DNN Frame', dnn_frame)
            cv2.imshow('ROI', roi)
            # cv2.imshow('Grayscale Frame', grayscale_frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

            self.frame_number += 1

        self.media.release()
        cv2.destroyAllWindows()

