import cv2


class Label:

    def __init__(self, related_object, title):

        self.related_object = related_object
        self.title = title
        self.lt_corner = None
        self.rb_corner = None

    def display_title(self, frame):

        # cx, cy = self.related_object.get_center_coordinates()
        cx = self.lt_corner[0]
        cy = self.lt_corner[1]

        cv2.putText(frame, str(self.title), (cx, cy + 14), 0, 0.5, (0, 0, 255), 1)

    def display_border(self, frame, border_height=20):

        self.lt_corner = (self.related_object.x, self.related_object.y - border_height)
        self.rb_corner = (self.related_object.x + self.related_object.width, self.related_object.y)

        cx, cy = self.related_object.get_center_coordinates()

        return cv2.rectangle(frame, self.lt_corner, self.rb_corner, self.related_object.color, self.related_object.thickness)