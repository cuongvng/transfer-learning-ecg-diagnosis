import sys
sys.path.append('../')
import torch
torch.manual_seed(0)
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from ribero.ribero_dataset import RiberoDataset
from cpsc.cpsc_dataset import CPSC2018
from georgia.georgia_dataset import GeorgiaDataset
from ptb.ptb_dataset import PtbDataset
from ptbxl.ptbxl_dataset import PTBXL
from resnet1d18 import ResNet1d18
from resnet50 import ResNet1d50
from resnet1d101 import ResNet1d101
from gru import GRU_Classifier
from lstm import LSTM, BiLSTM
from pretrain.eval import classify, get_f1

DATASETS = {
    'ribero': RiberoDataset,
    'cpsc': CPSC2018,
    'georgia': GeorgiaDataset,
    'ptb': PtbDataset,
    'ptbxl': PTBXL
}

RESNETs = {
    'resnet1d18': ResNet1d18,
    'resnet1d50': ResNet1d50,
    'resnet1d101': ResNet1d101,
}

RNN = {
    'gru': GRU_Classifier,
    'lstm': LSTM,
    'bilstm': BiLSTM
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def train_from_scratch(model, model_name, train_data, test_data):
    batch_size = 256
    weights = test_data.y.sum(axis=0)/len(test_data)
    best_avg_f1 = 0.0

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=None)

    model_dir = './checkpoints/scratch'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    start = datetime.now()
    filename = os.path.join(model_dir, f'./{model_name}_{downstream_data}_scratch.pt')
    log_file = os.path.join(model_dir, f'./{model_name}_{downstream_data}_scratch.csv')
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


if __name__ == '__main__':
    for downstream_data, dataset_instance in DATASETS.items():
        train_data = dataset_instance("train", data_dir=f"{downstream_data}/data")
        test_data = dataset_instance("test", data_dir=f"{downstream_data}/data")

        for model_name, resnet_instance in RESNETs.items():
            model = resnet_instance(num_classes=len(train_data.CLASSES)).to(device=device, dtype=torch.double)
            print(f"Train {model_name} on {downstream_data} from scratch...")
            train_from_scratch(model, model_name, train_data, test_data)

        for model_name, rnn_instance in RNN.items():
            model = rnn_instance(num_classes=len(train_data.CLASSES), device=device).to(device=device, dtype=torch.double)
            model.device = device
            print(f"Train {model_name} on {downstream_data} from scratch...")
            train_from_scratch(model, model_name, train_data, test_data)