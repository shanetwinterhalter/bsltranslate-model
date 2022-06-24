from csv import reader, writer
from errno import ENOENT, ENOENT
from hashlib import md5
from os import strerror
from os.path import isfile
from pickle import dump, load


def read_from_file(filename, delim=' ', delete_headers=False):
    '''Read dataset from file'''
    data = {}
    with open(filename, newline='') as f:
        csv_data = reader(f, delimiter=delim)
        for idx, row in enumerate(csv_data):
            data[idx] = row
    if delete_headers:
        del data[0]
    return data


def check_file_exists(filename):
    '''Check whether a file exists'''
    if not isfile(filename):
        raise FileNotFoundError(ENOENT,
                                strerror(ENOENT),
                                filename)


def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as pkl_file:
        output = load(pkl_file)
    return output


def save_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        dump(data, pkl_file)


def calc_md5(path):
    hash_md5 = md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_file_duplicates(video_dir, file_list):
    hash_list = []
    for file in file_list:
        vid_hash = calc_md5(video_dir + "/" + file)
        if vid_hash in hash_list:
            raise Exception("Duplicate file found: " + file +
                            ". Remove this before continuing")
        else:
            hash_list.append(vid_hash)


def save_csv(var, output_filename):
    '''Save dataset to csv file'''
    with open(output_filename, 'w', newline='') as csv_file:
        file = writer(csv_file, delimiter=',')
        if type(var) == dict:
            for key, value in var.items():
                file.writerow([key, ''.join(value)])
        elif type(var) == list or type(var) == tuple:
            for i in var:
                file.writerow(i)
