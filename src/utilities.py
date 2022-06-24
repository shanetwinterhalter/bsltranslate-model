from math import log10
from matplotlib.pyplot import figure, imshow, show
from numpy import exp

from .training import get_optimizer


def lr_finder(dl, model, loss_fn, weight_decay=0.0,
              init_value=1e-8, final_value=1., beta=0.98):
    optimizer = get_optimizer(model, lr=init_value, wd=weight_decay)
    num = len(dl)-1
    mult = (final_value / init_value)**(1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for x, y in dl:
        batch_num += 1
        # Get the loss for the batch
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1-beta**batch_num)
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        losses.append(smoothed_loss)
        log_lrs.append(log10(lr))
        loss.backward()
        optimizer.step()
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


def convert_to_color_image(new_df):
    data_list = new_df.values.tolist()
    for frame_idx, frame in enumerate(data_list):
        new_frame = [] * int(len(frame) / 3)
        for idx in range(0, len(frame), 3):
            new_frame.append([frame[idx], frame[idx + 1], frame[idx + 2]])
        data_list[frame_idx] = new_frame
    return data_list


def sigmoid(df):
    for col in df:
        df[col] = 1/(1 + exp(-df[col]))
    return df


def plot_key(key, data_labels):
    for sign in data_labels[key]:
        sigmoid_sign = sigmoid(sign)
        data_list = convert_to_color_image(sigmoid_sign)
        figure()
        imshow(data_list)
    show()


def save_to_csv(data, file_name, headers=None):
    with open(file_name, 'w') as f:
        if headers is not None:
            f.write(','.join(headers) + '\n')
        for frame in data:
            f.write(','.join(map(str, frame)) + '\n')
