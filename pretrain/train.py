import torch
torch.manual_seed(0)
import torch.optim as optim
import os
from datetime import datetime
from ptbxl.ptbxl_dataset import PTBXL
from cpsc.cpsc_dataset import CPSC2018
from georgia.georgia_dataset import GeorgiaDataset
from resnet1d101 import ResNet1d101
from resnet50 import ResNet1d50
from resnet1d18 import ResNet1d18
from lstm import LSTM, BiLSTM
from gru import GRU_Classifier
from transformer import EcgTransformer
from eval import classify, get_f1


def train(model_name, data_dir, model_instance, device, train_data, val_data=None):
    print("device:", device)

    batch_size = 256

    model = model_instance.to(device=device, dtype=torch.double)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=None)

    model_dir = os.path.join(data_dir, './checkpoints')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    start = datetime.now()
    filename = os.path.join(model_dir, f'./{model_name}_{data_dir[2:]}_{start.isoformat()}.pt')
    f = open(os.path.join(model_dir, f'./log_{model_name}_{data_dir[2:]}_{start.isoformat()}.txt'), 'a')

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    best_f1 = 0.0
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
                f.write(log+'\n')

            # Free up GPU memory
            if device == torch.device('cuda'):
                del data, labels, y_hat
                torch.cuda.empty_cache()

        if val_data:
            # Eval on val set
            y_trues_val, y_preds_val= classify(model, device, val_data)
            f1_val = get_f1(y_trues_val, y_preds_val)
            print("F1 val:", f1_val.round(4))
            f.write(f"F1 val: {f1_val.round(4)}\n")


            # Save checkpoint:
            if f1_val.mean() > best_f1:
                best_f1 = f1_val.mean()
                print(f"Best mean f1 scores = {f1_val.mean():.2f}, saving checkpoint ...")

                checkpoint = {
                    'model': model,
                    'state_dict': model.state_dict(),
                }
            
                torch.save(checkpoint, filename)
                print("Checkpoint saved!")
                f.write("Checkpoint saved!\n")

        else:
            if epoch % 10 == 0:
                checkpoint = {
                    'model': model,
                    'state_dict': model.state_dict(),
                }
            
                torch.save(checkpoint, filename)
                print("Checkpoint saved!")


    end = datetime.now()
    print("Training time:", end - start)
    f.close()

def main():
    data_name = 'ptbxl'
    data_dir = f'./{data_name}'
    train_data = PTBXL('train', f'./{data_name}/data/')
    val_data = PTBXL('test', f'./{data_name}/data/')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = 'resnet1d101'
    # model_instance = ResNet1d101(num_classes=len(train_data.CLASSES))

    # model_name = 'resnet1d50'
    # model_instance = ResNet1d50(num_classes=len(train_data.CLASSES))

    # model_name = 'resnet1d18'
    # model_instance = ResNet1d18(num_classes=len(train_data.CLASSES))

    model_name = 'lstm'
    model_instance = LSTM(num_classes=len(train_data.CLASSES),device=device)

    # model_name = 'bilstm'
    # model_instance = BiLSTM(num_classes=len(train_data.CLASSES),device=device)

    # model_name = 'gru'
    # model_instance = GRU_Classifier(num_classes=len(train_data.CLASSES),device=device)

    # model_name = 'transformer'
    # model_instance = EcgTransformer(num_classes=len(train_data.CLASSES))

    train(model_name, data_dir, model_instance, device, train_data, val_data)

if __name__ ==  '__main__':
    main()
