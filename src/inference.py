import mediapipe as mp

from cv2 import COLOR_BGR2RGB, CAP_PROP_POS_MSEC, COLOR_RGB2BGR, cvtColor,  \
    destroyAllWindows, imshow, putText, waitKey, VideoCapture
from torch import argmax, cat, empty, float, float64, tensor

from .data.text_variables import *
from .datasets import convert_tensor_to_3d
from .mediapipe import convert_mp_to_tensor


def normalize_frame(frame, mean, std):
    '''Normalizes the frame'''
    return (frame - mean) / std


def load_video(file):
    if file is None:
        return VideoCapture(0)
    else:
        return VideoCapture(file)


def process_frame(hands, frame, mean, std, no_coords=126, debug=False):
    '''Gets hand coordinates for single video frame'''
    frame = cvtColor(frame, COLOR_BGR2RGB)
    mp_result = hands.process(frame)
    if mp_result.multi_hand_world_landmarks:
        data = normalize_frame(convert_mp_to_tensor(mp_result), mean, std)
    else:
        data = empty((1, 1, no_coords), dtype=float)

    if debug:
        return data, mp_result
    return data


def get_prediction(cur_sign, frames_per_sign, model):
    reshaped_tensor = convert_tensor_to_3d(cur_sign, frames_per_sign)
    out = model(reshaped_tensor.unsqueeze(0))
    return argmax(out, dim=1)


def detokenise(vocab, pred):
    '''Converts the prediction into a string'''
    return vocab[int(pred)]


def predict_signs(video, model, vocab, normalization_stats,
                  frames_per_sign=30, number_of_coords=126,
                  display_pred=False, torchscript_model=False):
    cur_sign = empty((1, frames_per_sign, number_of_coords), dtype=float64)
    mean = tensor(normalization_stats[0])
    std = tensor(normalization_stats[1])
    predictions = []
    with mp.solutions. \
        hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
        while video.isOpened():
            success, frame = video.read()
            if success:
                new_frame = process_frame(hands, frame, mean, std,
                                          number_of_coords)
                cur_sign = cat((cur_sign[:, 1:, :], new_frame), dim=1)
                if torchscript_model:
                    cur_sign = cur_sign.float()
                pred_token = get_prediction(cur_sign, frames_per_sign, model)
                pred = detokenise(vocab, pred_token)
                predictions.append([video.get(CAP_PROP_POS_MSEC), pred])
                if display_pred:
                    if pred != 'NaS':
                        putText(frame, pred, bottomLeftCornerOfText, font,
                                fontScale, fontColor, thickness, lineType)
                    imshow("Frame", frame)
            else:
                break
            # Terminate when pressing esc
            if waitKey(5) & 0xFF == 27:
                break
        video.release()
        destroyAllWindows()
    return predictions


def plot_hand_coords(mp_result, frame):
    # Plot coords onto image for debugging
    frame.flags.writeable = True
    frame = cvtColor(frame, COLOR_RGB2BGR)
    if mp_result.multi_hand_landmarks:
        for hand_landmarks in mp_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.
                get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.
                get_default_hand_connections_style())
    return frame
