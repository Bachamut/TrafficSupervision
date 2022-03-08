import numpy as np


class PositionHistory:

    def __init__(self, related_object, frame_number):

        self.related_object = related_object
        self.position_history = np.array([frame_number, self.related_object.center_position], dtype=object)

    def update_position_history(self, frame_number):

        new_entry = np.array([frame_number, self.related_object.center_position], dtype=object)
        updated_history = np.vstack([self.position_history, new_entry])
        self.position_history = updated_history.copy()
