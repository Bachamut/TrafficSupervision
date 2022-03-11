import cv2


def frame_content_handler(frame, od, detection_handler):

    (class_ids, scores, boxes) = od.detect(frame)

    for entity, status in detection_handler.tracked_entity.items():

        cx, cy = entity.box.get_center_coordinates()

        create_text_label(frame, str(entity.entity_id.number), cx, cy)

        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # confidence = round(score, 4) * 100
        # print(f'{class_name} - score {"{:.2f}".format(confidence)}%')


def create_text_label(frame, text, cx, cy):
    cv2.putText(frame, text, (cx, cy), 0, 0.5, (0, 0, 255), 1)


def display_frame_number(frame, frame_number):
    cv2.putText(frame, ("Frame No." + str(frame_number)), (30, 30), 0, 0.5, (0, 0, 255), 1)


