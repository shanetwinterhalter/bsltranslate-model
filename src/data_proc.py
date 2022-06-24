from pandas import DataFrame


def normalize_data(data):
    new_df = DataFrame(data)
    col_mean = []
    col_std = []
    # Normalize each column
    for column in new_df:
        col_mean.append(new_df[column].mean())
        col_std.append(new_df[column].std())
        new_df[column] = (new_df[column] - col_mean[-1]) / col_std[-1]
    normalization_stats = (col_mean, col_std)
    return new_df, normalization_stats


def tokenize(label, vocab):
    for k, v in vocab.items():
        if v[0] == label:
            return k
