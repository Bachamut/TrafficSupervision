import cv2


class FramePreparation:

    def __init__(self, threshhold_config=(20, 255, cv2.THRESH_BINARY), gaussian_blur_config=((3, 5), cv2.BORDER_DEFAULT)):

        self.threshhold_config = threshhold_config
        self.gaussian_blur_config = gaussian_blur_config

    def prepare_frame(self, frame):

        # frame = self.frame_resize(frame)
        # grayclace_frame = self.convert_grayscale(frame)
        # _, background_thresh = self.apply_threshold(grayclace_frame)

        return frame

    def apply_threshold(self, background_mask):

        thresh = self.threshhold_config[0]
        max_val = self.threshhold_config[1]
        mode = self.threshhold_config[2]

        _, background_thresh = cv2.threshold(background_mask, thresh, max_val, mode)

        return _, background_thresh

    def apply_blur(self, frame):

        k_size = self.gaussian_blur_config[0]
        border_type = self.gaussian_blur_config[1]

        blurred_frame = cv2.GaussianBlur(frame, k_size, border_type)

        return blurred_frame

    def convert_grayscale(self, frame):

        grayclace_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return grayclace_frame

    def frame_resize(self, frame):

        resized_frame = cv2.resize(frame, (0, 0), None, .5, .5)

        return resized_frame
