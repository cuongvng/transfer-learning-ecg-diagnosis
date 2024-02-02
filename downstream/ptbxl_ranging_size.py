import sys
sys.path.append('../')
import torch
torch.manual_seed(0)
import numpy as np
import torch.optim as optim
import os
from datetime import datetime
from ptbxl.ptbxl_ranging_size_datasets import PTBXLPartition
from pretrain.eval import classify, get_f1
from resnet1d18 import ResNet1d18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def train_from_scratch(model_name, downstream_data, training_size):
    batch_size = 256
    train_data = PTBXLPartition("train", data_dir=f"{downstream_data}/data", training_size=training_size)

    test_data = PTBXLPartition("test", data_dir=f"{downstream_data}/data")

    weights = test_data.y.sum(axis=0)/len(test_data)
    best_avg_f1 = 0.0

    model = ResNet1d18(num_classes=len(train_data.CLASSES)).to(device=device, dtype=torch.double)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=None)

    model_dir = './checkpoints/ptbxl_ranging_size'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    start = datetime.now()
    filename = os.path.join(model_dir, f'./{model_name}_{downstream_data}_{training_size}_scratch.pt')
    log_file = os.path.join(model_dir, f'./{model_name}_{downstream_data}_{training_size}_scratch.csv')
    with open(log_file, 'w') as f:
        f.write(f"fold|f1|weights|avg_f1\n")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device, dtype=torch.double)

            optimizer.zero_grad()        
            y_hat = model(data)
            
            loss = loss_func(y_hat.float(), labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                log = "Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss)
                print(log)

            # Free up GPU memory
            if device == torch.device('cuda'):
                del data, labels, y_hat
                torch.cuda.empty_cache()

        # Eval on test set
        y_trues_val, y_preds_val= classify(model, device, test_data)
        f1_val = get_f1(y_trues_val, y_preds_val)
        avg_f1 = np.sum(f1_val * weights) / np.sum(weights)
        print("F1 val:", f1_val.round(4), "avg:", avg_f1)
        best_avg_f1 = max(best_avg_f1, avg_f1)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch}|{list(f1_val.round(4))}|{list(weights.round(4))}|{avg_f1.round(4)}\n")

    end = datetime.now()

    # Save checkpoint:
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
    }

    torch.save(checkpoint, filename)
    print("Checkpoint saved!")


    print("Training time:", end - start)
    f.close()

    print("Best avg F1:", best_avg_f1)


def _finetune(upstream_data, model_name, downstream_data, training_size):
    batch_size = 256

    train_data = PTBXLPartition("train", data_dir=f"{downstream_data}/data", training_size=training_size)

    test_data = PTBXLPartition("test", data_dir=f"{downstream_data}/data")

    weights = test_data.y.sum(axis=0)/len(test_data)
    best_avg_f1 = 0.0

    #### Load pretrained model
    pretrained_checkpoint = f"../pretrain/{upstream_data}/checkpoints/{model_name}_{upstream_data}.pt"
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])

    model.fc2 = torch.nn.Linear(in_features=512, out_features=len(train_data.CLASSES), bias=True)
    assert model.fc2.weight.size(0) == len(train_data.CLASSES)

    model.to(device=device, dtype=torch.double)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=None)

    model_dir = f'./checkpoints/ptbxl_ranging_size'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    start = datetime.now()
    filename = os.path.join(model_dir, f'./{model_name}_{upstream_data}_to_{downstream_data}_{training_size}.pt')
    log_file = os.path.join(model_dir, f'./{model_name}_{upstream_data}_to_{downstream_data}_{training_size}.csv')
    with open(log_file, 'w') as f:
        f.write(f"fold|f1|weights|avg_f1\n")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device, dtype=torch.double)

            optimizer.zero_grad()        
            y_hat = model(data)
            
            loss = loss_func(y_hat.float(), labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                log = "Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss)
                print(log)

            # Free up GPU memory
            if device == torch.device('cuda'):
                del data, labels, y_hat
                torch.cuda.empty_cache()

        # Eval on test set
        y_trues_val, y_preds_val= classify(model, device, test_data)
        f1_val = get_f1(y_trues_val, y_preds_val)
        avg_f1 = np.sum(f1_val * weights) / np.sum(weights)
        print("F1 val:", f1_val.round(4), "avg:", avg_f1)
        best_avg_f1 = max(best_avg_f1, avg_f1)

        with open(log_file, 'a') as f:
            f.write(f"{epoch}|{list(f1_val.round(4))}|{list(weights.round(4))}|{avg_f1.round(4)}\n")

    end = datetime.now()

    # Save checkpoint:
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
    }

    torch.save(checkpoint, filename)
    print("Checkpoint saved!")

    print("Training time:", end - start)
    print("Best avg F1:", best_avg_f1)


if __name__ == '__main__':
    model_name = 'resnet1d18'
    upstream_data = 'georgia'
    downstream_data = 'ptbxl'
    training_sizes = [2000, 4000, 6000, 8000, 10000, 12000]

    for training_size in training_sizes:
        print(f"Finetuning on {training_size} samples")
        _finetune(upstream_data, model_name, downstream_data, training_size)
        print(f"Train from scratch on {training_size} samples")
        train_from_scratch(model_name, downstream_data, training_size)
