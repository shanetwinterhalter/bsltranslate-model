import argparse

from datetime import datetime
from src.files import load_pickle, read_vocab_file
from src.models import Cnn_3d, load_model
from src.inference import load_video, predict_signs
from src.utilities import save_to_csv


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str,
                        default='models/'
                        '20220620212654/desktop/',
                        help='Path to the model folder')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file to process'),
    parser.add_argument('--write_to_file', type=bool, default=False,
                        help='Write predictions to file')
    args = parser.parse_args()
    return args


def main():
    options = get_command_line_args()
    model_name = Cnn_3d
    model_fname = "stream_cnn_3d"
    model_path = options.model_folder + model_fname + ".pt"
    normalization_stats_path = options.model_folder + model_fname \
        + "_norm_stats.pkl"
    vocab_file_path = options.model_folder + model_fname + "_vocab.csv"

    # Load the normalization stats
    normalization_stats = load_pickle(normalization_stats_path)

    # Load vocab
    vocab = read_vocab_file(vocab_file_path)

    # Load the model
    model = load_model(model_name, len(vocab), model_path)

    # Open the video
    video = load_video(options.video)

    # Loop over the frames of the video & process
    predictions = predict_signs(video, model, vocab, normalization_stats,
                                frames_per_sign=7, number_of_coords=126,
                                display_pred=True)

    if options.write_to_file:
        d_today = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_csv(predictions, 'inference_output_{}.csv'.format(d_today),
                    headers=['timestamp', 'prediction'])


if __name__ == '__main__':
    main()
