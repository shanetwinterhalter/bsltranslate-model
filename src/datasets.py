from torch import cat, tensor
from torch import float, flatten, unsqueeze, stack
from torch.nn.functional import pad


class DatasetFixed:
    def __init__(self, data, frames_per_sign=126):
        self.data = data
        self.frames_per_sign = frames_per_sign

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        data_label = tensor(self.data[i][1])
        sign_data = tensor(self.data[i][0].values, dtype=float)
        number_of_frames = sign_data.size()[0]
        if number_of_frames < self.frames_per_sign:
            sign_data = pad(sign_data,
                            (0, 0, 0, self.frames_per_sign - number_of_frames),
                            mode='constant', value=0.0)
        elif number_of_frames > self.frames_per_sign:
            # Take sign data from centre of sign
            sign_data = sign_data[
                int(number_of_frames/2) - int(self.frames_per_sign/2):
                int(number_of_frames/2) + int(self.frames_per_sign/2), :]
        # For linear model
        # return torch.flatten(sign_data), data_label
        # For cnn model
        # return torch.unsqueeze(sign_data, 0), data_label
        # For Resnet (requires 3 channels)
        return sign_data.repeat(3, 1, 1), data_label


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


# Treat each dimension as a separate channel.
# Output size should be (3 x frames x no_coords / 3)
class DatasetStreamMultiChannel(DatasetStream):
    def __init__(self, data, norm_stats, frames_per_sign=30,
                 label_col='label', metadata_cols=['timestamp', 'vid_fname']):
        super().__init__(data, norm_stats, frames_per_sign,
                         label_col, metadata_cols)

    def __getitem__(self, i):
        # TODO - If this works, include multichannel from start
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
        # Convert to multichannel
        x_coords = sign_data.flatten()[0::3].reshape(self.frames_per_sign, -1)
        y_coords = sign_data.flatten()[1::3].reshape(self.frames_per_sign, -1)
        z_coords = sign_data.flatten()[2::3].reshape(self.frames_per_sign, -1)
        output_data = stack((x_coords, y_coords, z_coords), dim=0)
        return output_data, tensor(self.labels[i])


# Treat each frame as a separate channel.
# Output size should be (frames, dims, coords)
class DatasetStreamFrameMultiChannel(DatasetStream):
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
        # Convert to multichannel
        x_coords = sign_data.flatten()[0::3].reshape(self.frames_per_sign, -1)
        y_coords = sign_data.flatten()[1::3].reshape(self.frames_per_sign, -1)
        z_coords = sign_data.flatten()[2::3].reshape(self.frames_per_sign, -1)
        output_data = stack((x_coords, y_coords, z_coords),
                            dim=1)
        return output_data, tensor(self.labels[i])


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
