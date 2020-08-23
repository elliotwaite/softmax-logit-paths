import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_POINTS_PER_CLASS = 20
NUM_STEPS = 1000
CLASS_COLORS = ('m', 'c', 'y')


class ZeroedLinear(nn.Linear):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


def get_model():
    return nn.Sequential(
        nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), ZeroedLinear(32, 3),
    )


def train_model(model, optimizer, inputs, labels):
    logits_at_each_step = []
    for _ in range(NUM_STEPS):
        logits = model(inputs)
        logits_at_each_step.append(logits.detach())
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return logits_at_each_step


def get_inputs_and_labels():
    x1 = torch.randn(NUM_POINTS_PER_CLASS, 3) + torch.tensor([[2, 0, 0]])
    x2 = torch.randn(NUM_POINTS_PER_CLASS, 3) + torch.tensor([[-2, 0, 0]])
    x3 = torch.randn(NUM_POINTS_PER_CLASS, 3)
    x3 /= torch.sum(x3 * x3, dim=1, keepdim=True).sqrt() / 4

    inputs = torch.cat((x1, x2, x3))
    labels = torch.cat(
        (
            torch.full((NUM_POINTS_PER_CLASS,), fill_value=0, dtype=torch.long),
            torch.full((NUM_POINTS_PER_CLASS,), fill_value=1, dtype=torch.long),
            torch.full((NUM_POINTS_PER_CLASS,), fill_value=2, dtype=torch.long),
        )
    )

    return inputs, labels


def plot_points(xs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for x, color in zip(xs, CLASS_COLORS):
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color)

    plt.show()


def plot_logit_paths(logits_at_each_step, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for logits, label in zip(torch.stack(logits_at_each_step).transpose(0, 1), labels):
        # Draws a line for each logit's path.
        ax.plot(
            logits[:, 0].numpy(),
            logits[:, 1].numpy(),
            logits[:, 2].numpy(),
            c=CLASS_COLORS[label.item()],
        )
        # Draws a point for each logit's final position.
        ax.scatter(
            logits[-1, 0].numpy(),
            logits[-1, 1].numpy(),
            logits[-1, 2].numpy(),
            c=CLASS_COLORS[label.item()],
        )

    plt.show()


def plot_input_points():
    inputs, labels = get_inputs_and_labels()
    x1, x2, x3 = inputs.split(NUM_POINTS_PER_CLASS)
    plot_points((x1, x2, x3))


def plot_output_logits_before_optimization():
    inputs, labels = get_inputs_and_labels()
    model = get_model()
    logits = model(inputs)
    y1, y2, y3 = logits.detach().split(NUM_POINTS_PER_CLASS)
    plot_points((y1, y2, y3))


def plot_output_logits_after_training_with_sgd():
    inputs, labels = get_inputs_and_labels()
    model = get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logits_at_each_step = train_model(model, optimizer, inputs, labels)
    plot_logit_paths(logits_at_each_step, labels)


def plot_output_logits_after_training_with_adam():
    inputs, labels = get_inputs_and_labels()
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters())
    logits_at_each_step = train_model(model, optimizer, inputs, labels)
    plot_logit_paths(logits_at_each_step, labels)


def main():
    # Uncomment any of the functions below to plot the data described by
    # the function's name.

    # plot_input_points()
    # plot_output_logits_before_optimization()
    plot_output_logits_after_training_with_sgd()
    # plot_output_logits_after_training_with_adam()


if __name__ == '__main__':
    main()
