from math import cos, pi
from random import seed, shuffle
from torch import argmax, bincount, cuda, device, float, tensor
from torch.nn.functional import pad
from torch.optim import AdamW


def split_by_video(df, validation_pct, seed_no=None):
    '''Split data into training and validation sets
    based on video filename'''
    video_list = df['vid_fname'].unique()
    split_point = round(len(video_list)*validation_pct)
    if seed_no:
        seed(seed_no)
    shuffle(video_list)
    train_vids = video_list[split_point:]
    val_vids = video_list[:split_point]
    train_df = df[df['vid_fname'].isin(train_vids)]
    valid_df = df[df['vid_fname'].isin(val_vids)]
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def split_list_by_pct(data_list, validation_pct, seed_no=None):
    split_point = round(len(data_list)*validation_pct)
    if seed_no:
        seed(seed_no)
    shuffle(data_list)
    return data_list[split_point:], data_list[:split_point]


def get_default_device():
    if cuda.is_available():
        return device('cuda')
    else:
        return device('cpu')


def to_device(data, device=None):
    if device is None:
        device = get_default_device()
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device=get_default_device()):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def get_optimizer(model, lr, wd):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = AdamW(parameters, lr=lr, weight_decay=wd)
    return optim


# Returns a list of learning rates to use
def set_learning_rates(total_batches, base_lr):
    pct_start, div_start = 0.25, 10
    lrs = [None]*total_batches
    for i in range(total_batches):
        pct = i/total_batches
        if pct < pct_start:
            pct /= pct_start
            lrs[i] = (1-pct)*base_lr/div_start + pct*base_lr
        else:
            pct = (pct-pct_start)/(1-pct_start)
            lrs[i] = 0.5*base_lr*cos((pct + base_lr)*pi) + 0.5*base_lr
    return lrs


def train_model(model, optim, loss_fn, train_dl,
                lrs, epoch):
    model.train()
    total = 0
    sum_loss = 0
    for idx, (x, y) in enumerate(train_dl):
        optim.param_groups[0]['lr'] = lrs[idx + epoch * len(train_dl)]
        batch = y.shape[0]
        output = model(x)
        loss = loss_fn(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
    return sum_loss/total


def valid_model(model, valid_dl, loss_fn, no_y_vals):
    model.eval()
    total = [0] * no_y_vals
    sum_loss = 0
    sign_correct = [0] * no_y_vals
    for x, y in valid_dl:
        bs = y.shape[0]
        out = model(x)
        loss = loss_fn(out, y)
        sum_loss += bs*(loss.item())
        pred = argmax(out, dim=1)
        total = [x + y for x, y in
                 zip(total, bincount(y, minlength=no_y_vals).tolist())]
        for correct_sign_idx in range(no_y_vals):
            cur_preds = (pred == correct_sign_idx)
            cur_labels = (y == correct_sign_idx)
            sign_correct[correct_sign_idx] += sum([cur_preds[i].item() and
                                                   cur_labels[i].item()
                                                   for i in range(bs)])
        # Exclude NaS from total_accuracy or it becomes most of the output
        total_accuracy = sum(sign_correct[1:]) / sum(total[1:]) \
            if sum(total[1:]) != 0 else 0
        correct_per_sign = [i / j if j != 0 else 0
                            for i, j in zip(sign_correct, total)]
    return sum_loss/sum(total), total_accuracy, correct_per_sign


def print_sign_stats(sign_correct, vocab):
    pct = [x[0]/x[1] if x[1] != 0 else 1 for x in sign_correct]
    for idx, _ in enumerate(sign_correct):
        print("Sign {} has accuracy {}".format(vocab[idx], pct[idx]))


def calc_class_weights(df, output_size):
    class_weights = tensor(1 / df['label'].value_counts().values,
                           dtype=float)
    # Because ResNet 18 has output size of 1000
    class_weights = pad(class_weights, (0, output_size - len(class_weights)),
                        'constant', 1e-25)
    return class_weights
