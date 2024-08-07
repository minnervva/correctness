import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import rc, rcParams

plt.rcParams['text.usetex'] = True
font = {'family' : 'sans-serif',
        'sans-serif': 'arial',
        'size'   : 16}

rc('font', **font)

cm = 1/2.54  # centimeters in inches
width = 8.86
rcParams['figure.figsize'] = (width, width/1.62)

class CustomSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(CustomSAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.agg_linear = nn.Linear(in_channels, out_channels, bias=False)
        
    def forward(self, x, edge_index):
        row, col = edge_index
        num_nodes = x.size(0)
        
        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        count = torch.zeros(num_nodes, x.size(1), device=x.device)

        sorted_index, sorted_indices = torch.sort(col)
        
        out.index_add_(dim=0, index=sorted_index, source=x[row[sorted_indices]])
        count.index_add_(dim=0, index=sorted_index, source=torch.ones_like(x[row[sorted_indices]]))

        out = out / count.clamp(min=1)
        
        out = self.agg_linear(out)
        out += self.linear(x)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out

class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.conv1 = CustomSAGEConv(dataset.num_node_features, 1024)
        self.conv2 = CustomSAGEConv(1024, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits

def run_training_loop(deterministic):
    if deterministic:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
        
    model = GraphSAGE().to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    weights_history = []
    metrics_history = []
    ratio_non_equal_history = []

    start_time = time.time()

    for epoch in range(epochs):
        loss = train(model, data, optimizer)
        accs, logits = test(model, data)
        train_acc, val_acc, test_acc = accs
        metrics_history.append((epoch + 1, loss, train_acc, val_acc, test_acc))
        weights_history.append((epoch + 1, [p.clone().cpu().detach().numpy() for p in model.parameters()]))
        if epoch > 0:
            non_equal_ratio = calculate_non_equal_ratio(weights_history[-2][1], weights_history[-1][1])
            ratio_non_equal_history.append(non_equal_ratio)
        else:
            ratio_non_equal_history.append(0)  # No comparison for the first epoch
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Non-equal Ratio: {ratio_non_equal_history[-1]:.4f}')
    
    end_time = time.time()
    duration = end_time - start_time

    return weights_history, metrics_history, ratio_non_equal_history, model, duration

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE().to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
epochs = 10
checkpoint_epochs = list(range(1, epochs + 1))

# Function to calculate relative difference for lists of weights
def relative_difference(weights1, weights2):
    differences = [np.linalg.norm(w1 - w2) / np.linalg.norm(w1) for w1, w2 in zip(weights1, weights2)]
    return np.mean(differences)

# Function to calculate the ratio of non-equal elements in tensors
def calculate_non_equal_ratio(weights1, weights2):
    non_equal_elements = sum(np.count_nonzero(w1 != w2) for w1, w2 in zip(weights1, weights2))
    total_elements = sum(w1.size for w1 in weights1)
    return non_equal_elements / total_elements

# Measure inference latency and ratio of non-equal elements
def measure_inference_latency_and_ratio(model, reference_weights, data, num_runs=100):
    latencies = []
    ratio_non_equal = []
    for _ in range(num_runs):
        start_time = time.time()
        model(data)
        end_time = time.time()
        latencies.append(end_time - start_time)

        current_weights = [p.clone().cpu().detach().numpy() for p in model.parameters()]
        ratio_non_equal.append(calculate_non_equal_ratio(reference_weights, current_weights))
    
    return np.mean(latencies), np.std(latencies), np.mean(ratio_non_equal), np.std(ratio_non_equal)

# Run deterministic training
det_weights, det_metrics, det_ratio_non_equal, det_model, det_duration = run_training_loop(deterministic=True)
print(f'Deterministic training duration: {det_duration:.2f} seconds')

# Measure inference latency and ratio for deterministic model
det_inference_mean, det_inference_std, det_ratio_non_equal_mean, det_ratio_non_equal_std = measure_inference_latency_and_ratio(
    det_model, det_weights[-1][1], data)
print(f'Deterministic inference latency: Mean: {det_inference_mean:.6f} seconds, Std Dev: {det_inference_std:.6f} seconds')
print(f'Deterministic inference ratio non-equal: Mean: {det_ratio_non_equal_mean:.6f}, Std Dev: {det_ratio_non_equal_std:.6f}')

# Run multiple non-deterministic trainings
non_det_weights_runs = []
non_det_metrics_runs = []
non_det_ratio_non_equal_runs = []
non_det_models = []
non_det_durations = []
non_det_inference_latencies = []
non_det_inference_ratios = []

for run in range(100):
    print(f'Non-Deterministic Run {run+1}')
    weights, metrics, ratio_non_equal, non_det_model, duration = run_training_loop(deterministic=False)
    non_det_weights_runs.append(weights)
    non_det_metrics_runs.append(metrics)
    non_det_ratio_non_equal_runs.append(ratio_non_equal)
    non_det_models.append(non_det_model)
    non_det_durations.append(duration)
    
    # Measure inference latency and ratio for non-deterministic model
    inference_mean, inference_std, ratio_non_equal_mean, ratio_non_equal_std = measure_inference_latency_and_ratio(
        non_det_model, weights[-1][1], data)
    non_det_inference_latencies.append((inference_mean, inference_std))
    non_det_inference_ratios.append((ratio_non_equal_mean, ratio_non_equal_std))

print(f'Non-deterministic training mean duration: {np.mean(non_det_durations):.2f} seconds, std dev: {np.std(non_det_durations):.2f} seconds')

# Compute mean and std deviation of non-deterministic inference latencies and ratios
non_det_inference_means = [latency[0] for latency in non_det_inference_latencies]
non_det_inference_stds = [latency[1] for latency in non_det_inference_latencies]
non_det_ratio_means = [ratio[0] for ratio in non_det_inference_ratios]
non_det_ratio_stds = [ratio[1] for ratio in non_det_inference_ratios]

print(f'Non-deterministic inference latency: Mean: {np.mean(non_det_inference_means):.6f} seconds, Std Dev: {np.std(non_det_inference_stds):.6f} seconds')
print(f'Non-deterministic inference ratio non-equal: Mean: {np.mean(non_det_ratio_means):.6f}, Std Dev: {np.std(non_det_ratio_stds):.6f}')

# Collect relative differences over epochs for plotting
relative_differences = {
    epoch: [
        relative_difference(
            det_weights[epoch - 1][1], 
            non_det_weights[epoch - 1][1]
        ) for non_det_weights in non_det_weights_runs
    ] for epoch in checkpoint_epochs
}

# Collect ratio of non-equal elements over epochs for plotting
ratio_non_equal = {
    epoch: [
        calculate_non_equal_ratio(
            det_weights[epoch - 1][1], 
            non_det_weights[epoch - 1][1]
        ) for non_det_weights in non_det_weights_runs
    ] for epoch in checkpoint_epochs
}

# Compute mean and std dev for train, val, test accuracy and loss
def calculate_metrics_statistics(metrics_runs, epochs):
    metrics_dict = {epoch: {'train_acc': [], 'val_acc': [], 'test_acc': [], 'loss': []} for epoch in epochs}

    for run in metrics_runs:
        for epoch, loss, train_acc, val_acc, test_acc in run:
            metrics_dict[epoch]['train_acc'].append(train_acc)
            metrics_dict[epoch]['val_acc'].append(val_acc)
            metrics_dict[epoch]['test_acc'].append(test_acc)
            metrics_dict[epoch]['loss'].append(loss)
    
    metrics_stats = {epoch: {} for epoch in epochs}
    
    for epoch in epochs:
        for metric in ['train_acc', 'val_acc', 'test_acc', 'loss']:
            values = metrics_dict[epoch][metric]
            metrics_stats[epoch][metric] = {
                'mean': np.mean(values),
                'std_dev': np.std(values)
            }
    
    return metrics_stats

metrics_stats = calculate_metrics_statistics(non_det_metrics_runs, checkpoint_epochs)

for epoch, stats in metrics_stats.items():
    print(f'Epoch {epoch}:')
    for metric, values in stats.items():
        print(f'  {metric.capitalize()}: Mean: {values["mean"]:.4f}, Std Dev: {values["std_dev"]:.4f}')

# Extracting the mean and std dev of train, val, test accuracy and loss
train_acc_means = [metrics_stats[epoch]['train_acc']['mean'] for epoch in checkpoint_epochs]
train_acc_stds = [metrics_stats[epoch]['train_acc']['std_dev'] for epoch in checkpoint_epochs]

val_acc_means = [metrics_stats[epoch]['val_acc']['mean'] for epoch in checkpoint_epochs]
val_acc_stds = [metrics_stats[epoch]['val_acc']['std_dev'] for epoch in checkpoint_epochs]

test_acc_means = [metrics_stats[epoch]['test_acc']['mean'] for epoch in checkpoint_epochs]
test_acc_stds = [metrics_stats[epoch]['test_acc']['std_dev'] for epoch in checkpoint_epochs]

loss_means = [metrics_stats[epoch]['loss']['mean'] for epoch in checkpoint_epochs]
loss_stds = [metrics_stats[epoch]['loss']['std_dev'] for epoch in checkpoint_epochs]

# Extracting the mean and std dev of relative differences
rel_diff_means = [np.mean(relative_differences[epoch]) for epoch in checkpoint_epochs]
rel_diff_stds = [np.std(relative_differences[epoch]) for epoch in checkpoint_epochs]

# Extracting the mean and std dev of ratio non-equal elements
ratio_non_equal_means = [np.mean(ratio_non_equal[epoch]) for epoch in checkpoint_epochs]
ratio_non_equal_stds = [np.std(ratio_non_equal[epoch]) for epoch in checkpoint_epochs]

# Plotting Mean Relative Difference
# plt.figure(figsize=(12, 8))
plt.figure()
plt.errorbar(checkpoint_epochs, rel_diff_means, yerr=rel_diff_stds, marker='o', label='Mean Relative Difference')
plt.xlabel(r'Epochs')
plt.ylabel(r'$V$')
# plt.title('Mean Relative Difference vs Epochs')
# plt.legend()
plt.grid(True)
plt.savefig("./training_variability.png", bbox_inches='tight')

# Plotting Ratio of Non-Equal Elements
plt.figure()
plt.errorbar(checkpoint_epochs, ratio_non_equal_means, yerr=ratio_non_equal_stds, marker='o', label='Ratio of Non-Equal Elements')
plt.xlabel(f'Epochs')
plt.ylabel(r'$V_{c}$')
# plt.title('Ratio of Non-Equal Elements vs Epochs')
# plt.legend()
plt.grid(True)
plt.savefig("./training_variability_count.png", bbox_inches='tight')

# Plotting accuracies
plt.figure()

# Plot Train Accuracy
plt.errorbar(checkpoint_epochs, train_acc_means, yerr=train_acc_stds, marker='o', label='Train Accuracy')

# Plot Validation Accuracy
plt.errorbar(checkpoint_epochs, val_acc_means, yerr=val_acc_stds, marker='o', label='Validation Accuracy')

# Plot Test Accuracy
plt.errorbar(checkpoint_epochs, test_acc_means, yerr=test_acc_stds, marker='o', label='Test Accuracy')

plt.xlabel(r'Epochs')
plt.ylabel(r'Accuracy')
# plt.title('Accuracy vs Epochs')
# plt.legend()
plt.grid(True)
plt.savefig("./training_metrics.png", bbox_inches='tight')

# Plotting Loss
plt.figure()
plt.errorbar(checkpoint_epochs, loss_means, yerr=loss_stds, marker='o', label='Loss')
plt.xlabel(r'Epochs')
plt.ylabel(r'Loss')
# plt.title('Loss vs Epochs')
# plt.legend()
plt.grid(True)
plt.savefig("./train_loss.png", bbox_inches='tight')
