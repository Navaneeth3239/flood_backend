import json

import torch
from torch import nn
from torch.nn import functional as functional


SEED = 7319
torch.manual_seed(SEED)
torch.set_num_threads(2)
FEATURE_MIN = torch.tensor([-9.210340372, -11.512925465, 1.897119985])
FEATURE_MAX = torch.tensor([2.302585093, 0.693147181, 5.298317367])


class EdgeMessageNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_weights = nn.Parameter(torch.full((4, 3), -0.8))
        self.input_bias = nn.Parameter(torch.full((4,), 6.0))
        self.output_weights = nn.Parameter(torch.full((4,), -1.0))
        self.output_bias = nn.Parameter(torch.tensor(-20.0))

    def forward(self, features):
        encoded = 2 * (features - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN) - 1
        hidden = functional.relu(encoded @ functional.softplus(self.input_weights).T + self.input_bias)
        return hidden @ functional.softplus(self.output_weights) + self.output_bias


def examples(count):
    bed = torch.rand(count, 16) * 10
    depth = torch.exp(torch.rand(count, 16) * 10 - 9)
    spacing = 5 + torch.rand(count, 1) * 145
    roughness = 0.005 + torch.rand(count, 1) * 0.145
    edges = [(row * 4 + column, next_row * 4 + next_column)
             for row in range(4) for column in range(4)
             for next_row, next_column in [(row + 1, column), (row, column + 1)]
             if next_row < 4 and next_column < 4]
    source = torch.tensor([edge[0] for edge in edges])
    target = torch.tensor([edge[1] for edge in edges])
    head = bed + depth
    difference = head[:, source] - head[:, target]
    wet_depth = (torch.maximum(head[:, source], head[:, target]) - torch.maximum(bed[:, source], bed[:, target])).clamp(0.0001, 10)
    slope = (difference.abs() / spacing).clamp(0.00001, 2)
    features = torch.stack([wet_depth.log(), slope.log(), -roughness.log().expand_as(slope)], dim=-1)
    log_flux = (5 / 3) * features[..., 0] + 0.5 * features[..., 1] + features[..., 2]
    return features, log_flux, difference.sign(), source, target


def aggregate(messages, source, target):
    result = torch.zeros(messages.shape[0], 16)
    result.index_add_(1, source, -messages)
    result.index_add_(1, target, messages)
    return result


def main():
    features, labels, direction, source, target = examples(2048)
    model = EdgeMessageNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for iteration in range(1400):
        batch = torch.randint(0, 1792, (64,))
        prediction = model(features[batch])
        scale = labels[batch].exp().amax(dim=1, keepdim=True).clamp_min(0.000001)
        predicted_messages = prediction.exp() / scale * direction[batch]
        target_messages = labels[batch].exp() / scale * direction[batch]
        loss = functional.mse_loss(prediction, labels[batch]) + 0.1 * functional.mse_loss(
            aggregate(predicted_messages, source, target), aggregate(target_messages, source, target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(features[1792:])
        relative_error = (prediction.exp() / labels[1792:].exp() - 1).abs()
        mean_error = relative_error.mean().item()
        if mean_error > 0.005:
            raise RuntimeError(f"Held-out relative error too high: {mean_error}")
        payload = {
            "schema": "terrain-edge-message-v1", "seed": SEED,
            "training": "Synthetic 4x4 hydraulic graphs; Manning edge flux and conservative node aggregation. Not calibrated to observed floods.",
            "trainingGraphs": 1792, "validationGraphs": 256,
            "validationMeanRelativeError": mean_error,
            "validationP95RelativeError": relative_error.quantile(0.95).item(),
            "featureMin": FEATURE_MIN.tolist(), "featureMax": FEATURE_MAX.tolist(),
            "inputWeights": functional.softplus(model.input_weights).tolist(),
            "inputBias": model.input_bias.tolist(),
            "outputWeights": functional.softplus(model.output_weights).tolist(),
            "outputBias": model.output_bias.item(),
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
