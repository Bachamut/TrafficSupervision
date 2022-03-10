import cv2

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

        self.object_detection = ObjectDetection()
        self.entity_handler = EntityHandler()
        self.id_handler = IdHandler()

        self.frame_number = 1

    def main_loop(self):

        while True:
            ret, frame = self.media.read()
            if not ret:
                break

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

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(0)
            if key == 27:
                break

            self.frame_number += 1

            # print(f'ID List:')
            # for id in self.id_handler.ids_pool:
            #     print(f'ID.{id.number} ffd({id.date_of_creation}): {id}')

            # for entity in self.entity_handler.tracked_entity:
            #     if len(entity.entity_id_history) > 30:
            #         print(f'{entity.entity_id_history}')

        self.media.release()
        cv2.destroyAllWindows()

