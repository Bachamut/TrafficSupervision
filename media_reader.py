import cv2

import frame_content
from background_subtractor import BackgroundSubtractor
from box import Box
from object_detection import ObjectDetection
from entity import Entity
from id_handler import IdHandler
from entity_handler import EntityHandler


class MediaReader:

    def __init__(self):
        media_dir = "media/video/los_angeles.mp4"
        # media_dir = "media/video/highway.mp4"
        self.media = cv2.VideoCapture(media_dir)
        self.frame_number = 1

        self.object_detection = ObjectDetection()
        self.entity_handler = EntityHandler()
        self.id_handler = IdHandler()
        self.background_subtractor = BackgroundSubtractor()

        self.background = self.background_subtractor.create_background()

        # self.background = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

        self.height = self.media.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.media.get(cv2.CAP_PROP_FRAME_WIDTH)

    def main_loop(self):

        while True:
            ret, frame = self.media.read()
            if not ret:
                break
            blur = cv2.GaussianBlur(frame, (1, 3), cv2.BORDER_DEFAULT)
            cv2.imshow('Blur', blur)
            roi = frame[0:int(self.height), 0:int(self.width)]
            blur = blur[0:int(self.height), 0:int(self.width)]
            background_mask = self.background.apply(blur)

            _, background_thresh = cv2.threshold(background_mask, 180, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(background_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = list()

            for cnt in contours:
                # Calculate area and remove small objects
                area = cv2.contourArea(cnt)
                if area > 400:
                    cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    detections.append([x, y, w, h])

            for rect in detections:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                cv2.putText(roi, "str(id)", (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)

            dnn = False
            if dnn:
                class_ids, scores, boxes = self.object_detection.detect(frame)

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

                    entity.box.display_border(frame)
                    entity.box.label.display_border(frame)
                    entity.box.label.display_title(frame)
                    entity.display_id(frame)
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
            cv2.imshow("Frame", frame)
            cv2.imshow("Background Thresh", background_thresh)
            cv2.imshow("Background Mask", background_mask)

            key = cv2.waitKey(0)
            if key == 27:
                break

            self.frame_number += 1

        self.media.release()
        cv2.destroyAllWindows()

