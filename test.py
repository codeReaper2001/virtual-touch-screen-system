from tensorflow import keras
import numpy as np
from typing import List
import time
import cv2

import util

MODEL_PATH = "./model/sign_classifier/keypoint_classifier_new.h5"
classes = ["Thumbs Up", "OK", "ROCK", "YEAH", "PALM", "C"]

def flatten_data(lm_list: List[util.LmData]):
    res: List[float] = []
    for lm in lm_list:
        res.extend([lm.x, lm.y, lm.z])
    return res



def do_something(img:cv2.Mat):
    detect_result = detector.find_hands(img)
    if not detect_result:
        return
    lm_list = detect_result.get_hand_world_lm_list()
    data = flatten_data(lm_list)
    data = np.array(data)
    data = data[None]
    start = time.time()
    preds = model.predict(data)
    end = time.time()
    print("predict time:", end - start)
    idx = np.argmax(np.squeeze(preds))
    cv2.putText(img, classes[idx], (200, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    # print()

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    fps_cal = util.FPSCalculator()
    detector = util.HandDetector(maxHands=1)

    model = keras.models.load_model(MODEL_PATH)

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        
        do_something(img)
    
        # fps
        fps = fps_cal.get_fps()
        if fps:
            cv2.putText(img, str(int(fps)), (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("img", img)
        
        key = cv2.waitKey(5)
        if key == 27:
            break
