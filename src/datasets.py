from torch import cat, tensor
from torch import float, unsqueeze, stack


class DatasetStream:
    def __init__(self, data, norm_stats, frames_per_sign=30,
                 label_col='label', metadata_cols=['timestamp', 'vid_fname']):
        self.labels = data[label_col]
        self.metadata = data[metadata_cols]
        self.data = data.drop([label_col] + metadata_cols, axis=1)
        self.frames_per_sign = frames_per_sign
        self.norm_values = self.calculate_col_zero_vals(norm_stats)

    def calculate_col_zero_vals(self, norm_stats):
        vals = [- (mean / std) for mean, std in zip(norm_stats[0],
                                                    norm_stats[1])]
        return vals

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        # TODO - Speed up
        if (i < self.frames_per_sign):
            sign_padding = tensor(self.norm_values, dtype=float).repeat(
                            (self.frames_per_sign - i - 1), 1)
            sign_data = tensor(self.data[0:i+1].values, dtype=float)
            sign_data = cat((sign_padding, sign_data), 0)
        elif (self.metadata['vid_fname'][i] !=
              self.metadata['vid_fname'][i-self.frames_per_sign]):
            # Find where the vid_fname changes and start from there
            # Then add padding for rest of the frames
            change_idx = self.metadata[self.metadata['vid_fname'] ==
                                       self.metadata['vid_fname'][i]].index[0]
            sign_data = tensor(self.data[change_idx:i+1].values, dtype=float)
            sign_padding = tensor(self.norm_values, dtype=float).repeat(
                                (self.frames_per_sign - i + change_idx - 1), 1)
            sign_data = cat((sign_padding, sign_data), 0)
        elif (i >= self.frames_per_sign):
            sign_data = tensor(self.data[i + 1 - self.frames_per_sign:i+1]
                               .values, dtype=float)
        return unsqueeze(sign_data, dim=0), tensor(self.labels[i])


def convert_tensor_to_3d(data, fps):
    # Convert to multichannel
    x_coords = data.flatten()[0::3].reshape(fps, -1)
    y_coords = data.flatten()[1::3].reshape(fps, -1)
    z_coords = data.flatten()[2::3].reshape(fps, -1)
    output_data = stack((x_coords, y_coords, z_coords),
                        dim=1)
    return unsqueeze(output_data, dim=0)


# 3d dataset with a single channel
# Output size should be (1, frames, dims, coords)
class DatasetStream3d(DatasetStream):
    def __init__(self, data, norm_stats, frames_per_sign=30,
                 label_col='label', metadata_cols=['timestamp', 'vid_fname']):
        super().__init__(data, norm_stats, frames_per_sign,
                         label_col, metadata_cols)

    def __getitem__(self, i):
        if (i < self.frames_per_sign):
            sign_padding = tensor(self.norm_values, dtype=float).repeat(
                            (self.frames_per_sign - i - 1), 1)
            sign_data = tensor(self.data[0:i+1].values, dtype=float)
            sign_data = cat((sign_padding, sign_data), 0)
        elif (self.metadata['vid_fname'][i] !=
              self.metadata['vid_fname'][i-self.frames_per_sign]):
            # Find where the vid_fname changes and start from there
            # Then add padding for rest of the frames
            change_idx = self.metadata[self.metadata['vid_fname'] ==
                                       self.metadata['vid_fname'][i]].index[0]
            sign_data = tensor(self.data[change_idx:i+1].values, dtype=float)
            sign_padding = tensor(self.norm_values, dtype=float).repeat(
                                (self.frames_per_sign - i + change_idx - 1), 1)
            sign_data = cat((sign_padding, sign_data), 0)
        elif (i >= self.frames_per_sign):
            sign_data = tensor(self.data[i + 1 - self.frames_per_sign:i+1]
                               .values, dtype=float)
        sign_data = convert_tensor_to_3d(sign_data, self.frames_per_sign)
        return sign_data, tensor(self.labels[i])
