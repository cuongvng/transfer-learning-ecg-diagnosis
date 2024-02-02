import sys
sys.path.append('../')
import torch
torch.manual_seed(0)
import numpy as np
import torch.optim as optim
import os
from datetime import datetime
from ribero.ribero_dataset import RiberoDataset
from cpsc.cpsc_dataset import CPSC2018
from georgia.georgia_dataset import GeorgiaDataset
from ptb.ptb_dataset import PtbDataset
from ptbxl.ptbxl_dataset import PTBXL
from pretrain.eval import classify, get_f1

DATASETS = {
    'ribero': RiberoDataset,
    'cpsc': CPSC2018,
    'georgia': GeorgiaDataset,
    'ptb': PtbDataset,
    'ptbxl': PTBXL
}

RESNETs = [
    'resnet1d18', 
    'resnet1d50', 
    'resnet1d101', 
]

RNNs = [
    'gru', 
    'lstm', 
    'bilstm'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def _finetune(upstream_data, model_name, downstream_data):
    batch_size = 256

    dataset_instance = DATASETS[downstream_data]
    train_data = dataset_instance("train", data_dir=f"{downstream_data}/data")
    test_data = dataset_instance("test", data_dir=f"{downstream_data}/data")

    weights = test_data.y.sum(axis=0)/len(test_data)
    best_avg_f1 = 0.0

    #### Load pretrained model
    pretrained_checkpoint = f"../pretrain/{upstream_data}/checkpoints/{model_name}_{upstream_data}.pt"
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])

    #### Replace the top layer to fit output
    if model_name in RNNs:
        model.device = device
        seq_len = 1000
        hidden_size = 100

        if model_name == 'gru' or model_name == 'lstm':
            model.fc = torch.nn.Linear(in_features=seq_len*hidden_size, out_features=len(train_data.CLASSES), bias=True)
            assert model.fc.weight.size(0) == len(train_data.CLASSES)
        elif model_name == 'bilstm':
            model.fc = torch.nn.Linear(in_features= 2 *seq_len*hidden_size, out_features=len(train_data.CLASSES), bias=True)
            assert model.fc.weight.size(0) == len(train_data.CLASSES)

    else: # resnets
        model.fc2 = torch.nn.Linear(in_features=512, out_features=len(train_data.CLASSES), bias=True)
        assert model.fc2.weight.size(0) == len(train_data.CLASSES)

    model.to(device=device, dtype=torch.double)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=None)

    model_dir = f'./checkpoints/upstream_{upstream_data}'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    start = datetime.now()
    filename = os.path.join(model_dir, f'./{model_name}_{upstream_data}_to_{downstream_data}.pt')
    log_file = os.path.join(model_dir, f'./{model_name}_{upstream_data}_to_{downstream_data}.csv')
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

def main():
    UPSTREAMS = [
        'ptbxl',
        'cpsc',
        'georgia'
    ]

    for upstream_data in UPSTREAMS:
        for downstream_data in DATASETS.keys():
            if downstream_data == upstream_data: # Exclude `upstream_data` from downstream datasets
                continue

            for model_name in RESNETs + RNNs:
                print(f"Finetuning on {downstream_data} with {model_name}_{upstream_data}...")
                _finetune(upstream_data, model_name, downstream_data)


if __name__ == '__main__':
    main()