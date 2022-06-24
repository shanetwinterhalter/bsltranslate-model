import mediapipe as mp

from cv2 import CAP_PROP_POS_MSEC, COLOR_BGR2RGB, cvtColor, flip, VideoCapture
from os.path import isfile
from torch import double, tensor, unsqueeze

from src.files import read_from_file, save_pickle


def get_headers():
    '''Returns a full list of headers for the Mediapipe coordinates'''
    headers = {}
    dims = ['x', 'y', 'z']
    hands = ['Left', 'Right']
    for hand in hands:
        for point in mp.solutions.hands.HandLandmark:
            for dim in dims:
                headers[hand + "_" + str(point) + "_" + dim] = []
    return headers


def convert_coords_to_list(mp_result, index):
    '''Converts the Mediapipe coordinates into a list'''
    no_dims = 3
    coords_per_hand = len(mp_result.multi_hand_world_landmarks[0].landmark)
    number_of_coords = coords_per_hand * no_dims
    coordinates = [0] * number_of_coords
    if index is not None:
        hand = mp_result.multi_hand_world_landmarks[index]
        for idx2, coord in enumerate(hand.landmark):
            coord_idx = idx2 * no_dims
            coordinates[coord_idx] = coord.x
            coordinates[coord_idx + 1] = coord.y
            coordinates[coord_idx + 2] = coord.z
    return coordinates


def get_hand_idx(hands):
    '''Figures out which coordinates belong to which hand. In case
    of multiple hands of same type (e.g. 2 left hands) assumes the
    higher probability is the real hand'''
    # This is awful, but need to account for hand labels being wrong
    no_hands = len(hands)
    if no_hands == 2:
        left_hand_prob = [None] * no_hands
        for idx, hand in enumerate(hands):
            label = hand.classification[0].label
            score = hand.classification[0].score
            if label == 'Right':
                score = 1 - score
            left_hand_prob[idx] = score
        # Choose hand with highest prob of being left as left
        # Other hand is assumed right
        max_prob = max(left_hand_prob)
        min_prob = min(left_hand_prob)
        left_hand_idx = left_hand_prob.index(max_prob)
        right_hand_idx = left_hand_prob.index(min_prob)
    elif no_hands == 1:
        hand = hands[0].classification[0].label
        if hand == "Left":
            left_hand_idx = 0
            right_hand_idx = None
        else:
            left_hand_idx = None
            right_hand_idx = 0
    else:
        left_hand_idx = None
        right_hand_idx = None
    return left_hand_idx, right_hand_idx


def convert_mp_to_dict(data, mp_result):
    '''Converts mediapipe output into a dictionary'''
    left_idx, right_idx = get_hand_idx(mp_result.multi_handedness)
    left_coord_list = convert_coords_to_list(mp_result, left_idx)
    right_coord_list = convert_coords_to_list(mp_result, right_idx)
    coordinates = left_coord_list + right_coord_list
    for idx, key in enumerate(data):
        data[key].append(coordinates[idx])
    return data


def convert_mp_to_tensor(mp_result):
    '''Converts mediapipe output into a tensor'''
    left_idx, right_idx = get_hand_idx(mp_result.multi_handedness)
    left_coord_list = convert_coords_to_list(mp_result, left_idx)
    right_coord_list = convert_coords_to_list(mp_result, right_idx)
    coordinates = tensor(left_coord_list + right_coord_list, dtype=double)
    coordinates = unsqueeze(coordinates, 0).unsqueeze(0)
    return coordinates


def empty_frame(data):
    for key in data:
        data[key].append(0)
    return data


def process_frame(data, hands, frame):
    '''Gets hand coordinates for single video frame'''
    frame = cvtColor(flip(frame, 1), COLOR_BGR2RGB)
    mp_result = hands.process(frame)
    if mp_result.multi_hand_world_landmarks:
        data = convert_mp_to_dict(data, mp_result)
    else:
        data = empty_frame(data)
    return data


# TODO - Implement a better algorithm here at some point
def get_label(labels, timestamp):
    # Check if timestamp is before first label or after last
    if timestamp < float(labels[1][0]) * 1000 or \
            timestamp > float(labels[len(labels)-1][1]) * 1000:
        return 'NaS'
    else:
        for key in labels:
            if key == 0:
                continue
            start_time = float(labels[key][0]) * 1000
            end_time = float(labels[key][1]) * 1000
            if timestamp >= start_time and timestamp <= end_time:
                return labels[key][2]
        return 'NaS'


def populate_metadata(metadata, video, vid_fname, labels):
    '''Adds frame metadata to a dict'''
    timestamp = int(video.get(CAP_PROP_POS_MSEC))
    metadata['timestamp'].append(timestamp)
    metadata['label'].append(get_label(labels, timestamp))
    metadata['vid_fname'].append(vid_fname)
    return metadata


def process_video(data, metadata, vid_fname, video_labels, confidence):
    '''Extracts each frame from the video and obtains hand
    coordinates which are appended to data'''
    video = VideoCapture(str(vid_fname))
    label_file = video_labels + "/" + \
        vid_fname.split("/")[-1].split(".")[0] + ".csv"
    labels = read_from_file(label_file, delim=',')
    with mp.solutions. \
         hands.Hands(static_image_mode=False,
                     max_num_hands=2,
                     min_detection_confidence=confidence) as hands:
        while(video.isOpened()):
            success, frame = video.read()
            if success:
                metadata = populate_metadata(metadata, video,
                                             vid_fname, labels)
                data = process_frame(data, hands, frame)
            else:
                break
    video.release()
    return data, metadata


# TODO - Could perhaps clean up this function
def process_file_list(file_list, video_dir, video_label_dir,
                      video_data_dir, confidence=0.5, overwrite_data=False):
    for video in file_list:
        file_path = video_dir + "/" + video
        output_file_path = video_data_dir + "/" + video + ".pkl"
        if isfile(output_file_path) and not overwrite_data:
            print("Skipping " + file_path + " as data already exists. Set "
                  "overwrite_data to True to overwrite")
        else:
            data = get_headers()
            metadata = {'timestamp': [], 'label': [], 'vid_fname': []}
            print("Processing {} and writing output "
                  "to {}".format(file_path, output_file_path))
            data, metadata = process_video(data, metadata, file_path,
                                           video_label_dir, confidence)
            save_pickle([data, metadata], output_file_path)
