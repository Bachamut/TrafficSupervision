import cv2

import label

# BGR
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


class Box:

    def __init__(self, dimensions, color=red, thickness=1):

        self.dimensions = dimensions
        (self.x, self.y, self.width, self.height) = self.dimensions
        self.color = color
        self.thickness = thickness
        self.related_entity = None
        self.label = None

    def display_border(self, frame):

        lt_corner = (self.x, self.y)
        rb_corner = (self.x + self.width, self.y + self.height)

        return cv2.rectangle(frame, lt_corner, rb_corner, self.color, self.thickness)

    def get_center_coordinates(self):

        center_x = int((self.x + self.width / 2))
        center_y = int((self.y + self.height / 2))

        return center_x, center_y

    def update_box_position(self, box):

        self.x = box.x
        self.y = box.y
        self.width = box.width
        self.height = box.height

    def create_label(self, title='new_label'):

        new_label = label.Label(self, title)

        return new_label

    def box_intersects(self, other):
        return not (self.top_right.x < other.bottom_left.x or
                    self.bottom_left.x > other.top_right.x or
                    self.top_right.y < other.bottom_left.y or
                    self.bottom_left.y > other.top_right.y)

