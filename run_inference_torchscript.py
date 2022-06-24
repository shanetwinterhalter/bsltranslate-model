import argparse
from torch import jit

from datetime import datetime
from src.files import load_pickle, read_from_file
from src.models import Cnn_3d, load_model
from src.inference import load_video, predict_signs
from src.utilities import save_to_csv


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='trained_models/'
                        '72_stream_cnn_resnet18_3d_20220508.pt',
                        help='Path to the model')
    parser.add_argument('--normalization_stats', type=str,
                        default='trained_models/normalization_stats_mhwl.pkl',
                        help='Path to normalization statistics file')
    parser.add_argument('--vocab_file', type=str,
                        default='vocab.csv',
                        help='Path to vocabulary file')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file to process'),
    parser.add_argument('--write_to_file', type=bool, default=False,
                        help='Write predictions to file')
    args = parser.parse_args()
    return args


def main():
    options = get_command_line_args()

    # Load the normalization stats
    normalization_stats = load_pickle(options.normalization_stats)

    # Load vocab
    vocab = read_from_file(options.vocab_file)

    # Load the model
    model = jit.load(options.model_path)

    # Open the video
    video = load_video(options.video)

    # Loop over the frames of the video & process
    predictions = predict_signs(video, model, vocab, normalization_stats,
                                frames_per_sign=30, number_of_coords=126,
                                display_pred=True, torchscript_model=True)

    if options.write_to_file:
        d_today = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_csv(predictions, 'inference_output_{}.csv'.format(d_today),
                    headers=['timestamp', 'prediction'])


if __name__ == '__main__':
    main()
