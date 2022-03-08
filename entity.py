import cv2

from position_history import PositionHistory


class Entity:

    def __init__(self, box, class_id=None, entity_id=None, score=None):

        self.entity_id = entity_id
        self.box = box
        self.class_id = class_id
        self.score = score
        self.last_detected: int = 0
        self.center_position = None
        self.position_history = None
        self.entity_id_history = list()
        self.predicted_position = None

        self.confidence_history = dict()

    def update_confidence_history(self, frame_number, class_id, score):

        self.confidence_history[frame_number] = [class_id, score]

    def attach_box(self, box):

        if not self.has_box():
            self.box = box
            box.related_entity = self.entity_id

    def has_box(self):

        if self.box is None:
            return False
        else:
            return True

    def update_entity_center_position(self, cx, cy):

        self.center_position = cx, cy
        return self.center_position

    def display_id(self, frame):

        ecx = self.center_position[0]
        ecy = self.center_position[1]
        cv2.putText(frame, str(self.entity_id.number), (ecx, ecy), 0, 0.5, (0, 255, 0), 1)

    def create_position_history(self, frame_number):

        self.position_history = PositionHistory(self, frame_number)

