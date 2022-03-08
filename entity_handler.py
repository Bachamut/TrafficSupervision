import numpy as np


class EntityHandler:

    def __init__(self):

        self.tracked_entity = dict()

    def find_convergent_entity(self, box):

        if len(self.tracked_entity) > 0:
            for entity in self.tracked_entity:
                if EntityHandler.distance_between(box, entity) <= 35:
                    return entity
            return None
        else:
            return None

    @staticmethod
    def distance_between(box, entity):

        fcx, fcy = box.get_center_coordinates()
        scx, scy = entity.box.get_center_coordinates()

        distance = ((((scx - fcx) ** 2) + ((scy - fcy) ** 2)) ** 0.5)

        return distance

    @staticmethod
    def update_previous_position(entity):

        entity.previous_frame_position = entity.center_position

    @staticmethod
    def distance_traveled(entity):

        pcx, pcy = entity.previous_frame_position
        ccx, ccy = entity.center_position

        distance = ((((pcx - ccx) ** 2) + ((pcy - ccy) ** 2)) ** 0.5)

        return distance

    @staticmethod
    def predict_entity_next_position(entity):

        if len(entity.position_history.position_history) > 1:
            last_position = np.array(entity.position_history.position_history[-2, 1])
            current_position = np.array(entity.position_history.position_history[-1, 1])
            displacement_vector = current_position - last_position
            predicted_position = current_position + displacement_vector

            return predicted_position
        else:
            print(f'Entity Id:{entity.entity_id.number} - there are only {len} entry in position history')

            return None
