import cv2
import csv
from typing import List

import util

cap = cv2.VideoCapture(1)
csv_path = 'model/sign_classifier/sign_new.csv'


def flatten_data(lm_list: List[util.LmData]):
    res: List[float] = []
    for lm in lm_list:
        res.extend([lm.x, lm.y, lm.z])
    return res


def do_something(img: cv2.Mat, detector: util.HandDetector, num:int, lmdata_generator:util.LmDataGenerator):
    detect_result = detector.find_hands(img)
    if not detect_result:
        return
    lm_list = detect_result.get_hand_world_lm_list()
    # print(lm_list[0])
    pre_process_data = flatten_data(lm_list)
    if num == -1:
        return
    enhanced_data = lmdata_generator.get_enhanced_data([pre_process_data], True)
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        for dataline in enhanced_data:
            writer.writerow([num, *dataline])


def key2num(key):
    if 48 <= key <= 57:
        number:int = key - 48
        return number
    else:
        return -1


def main():
    fps_cal = util.FPSCalculator()
    detector = util.HandDetector(maxHands=1)
    lmdata_generator = util.LmDataGenerator(rotate_range=30)
    while True:
        success, img = cap.read()
        if not success:
            break
        
        key = cv2.waitKey(5)
        if key == 27:
            break
        num = key2num(key)

        img = cv2.flip(img, 1)

        do_something(img, detector, num, lmdata_generator)

        # fps
        fps = fps_cal.get_fps()
        if fps:
            cv2.putText(img, str(int(fps)), (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("image", img)


if __name__ == "__main__":
    main()
