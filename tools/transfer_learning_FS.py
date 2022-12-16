

# Transfer Learning Dash1 server
# FF-patched not c40-patched
# working on FaceSwap first! 
# xception -> effnet -> mobilenet -> meso
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch import optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

import argparse
import random
import numpy as np
import os

from tqdm import tqdm

import PIL.Image as Image

import wandb


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
set_seed(2022)

parser = argparse.ArgumentParser(description='WDC2022')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--resize', default=128, type=int, help='transform size')
parser.add_argument('--dataname', default='DeepFake', type=str, help='data name')
parser.add_argument('--gpus', default='0,1', type=str, help='id of gpus to use')
parser.add_argument('--num_gpus', default=2, type=int, help='numbers of gpus to use')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus  

wandb.init(project="TL DeepFake",  entity="seanko96")


num_epochs = 40
lr = 0.001


wandb.config = {
    "learning_rate": lr,
    "epochs": num_epochs,
    "batch_size": args.batch_size,
}
wandb.run.name = args.dataname

transform_train = transforms.Compose([
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

TRAIN_PATH = "/media/data1/donggeun/FF_patched/FaceSwap/train/"
#TRAIN_PATH = "/media/data1/donggeun/c40_patched/DeepFake/train/"
VAL_PATH = '/media/data1/donggeun/masked_val/FaceForensics++/FaceSwap'
SAVE_PATH = '/home/donggeun/ckpt/patched/'
MODEL_NAME = 'xception'
WEIGHT_PATH = "/home/donggeun/ckpt/baseline/baseline_FaceSwap_xception_best_epoch_38.pth"
DATASET_NAME = VAL_PATH.split("/")[-1]

# train dataset
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform_train)

print(train_dataset.classes)
print(train_dataset.class_to_idx)

val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform_train)

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = 'entire_model.pth'
# model = torch.load(model_path).to(device)
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1).to(device)

model.load_state_dict(torch.load(WEIGHT_PATH))
model = nn.DataParallel(model, device_ids=range(args.num_gpus))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_val_f1 = 0
best_epoch = 0
best_val_acc = 0

for epoch in range(num_epochs):
    print(f'======EPOCH : {epoch}======')
    model.train()

    train_precision = []
    train_recall = []
    train_f1 = []

    val_loss = 0
    val_acc = 0
    val_precision = []
    val_recall = []
    val_f1 = []

    train_epoch_loss, val_epoch_loss = 0, 0
    train_epoch_acc, val_epoch_acc = 0, 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        inputs, targets = data
        
        # targets = targets.unsqueeze(1).type(torch.FloatTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        output = model(inputs).squeeze()
        loss = criterion(output, targets.to(torch.float32))
        preds = (output >= 0.5).float()
        
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        train_epoch_acc += (preds == targets).float().mean()
        
        # f1-score
        train_precision.append(precision_score(targets.to('cpu'), preds.to('cpu')))
        train_recall.append(recall_score(targets.to('cpu'), preds.to('cpu')))
        train_f1.append(f1_score(targets.to('cpu'), preds.to('cpu')))
        
    train_epoch_loss /= len(train_loader)
    train_epoch_acc /= len(train_loader)
        
    print('====EPOCH_LOSS : {:.4f}===='.format(train_epoch_loss))
    print('====EPOCH_ACC : {:.4f}===='.format(train_epoch_acc))
    print('====EPOCH_PRECISION : {:.4f}===='.format(np.mean(train_precision)))
    print('====EPOCH_RECALL : {:.4f}===='.format(np.mean(train_recall)))
    print('====EPOCH_F1 : {:.4f}===='.format(np.mean(train_f1)))

    wandb.log({"EPOCH_LOSS": train_epoch_loss})
    wandb.log({"EPOCH_ACCURACY": train_epoch_acc})
    wandb.log({"EPOCH_PRECISION": np.mean(train_precision)})
    wandb.log({"EPOCH_RECALL": np.mean(train_recall)})
    wandb.log({"EPOCH_F1": np.mean(train_f1)})

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader)):
            inputs, targets = data

            # targets = targets.unsqueeze(1).type(torch.FloatTensor)
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs).squeeze()
            loss = criterion(output, targets.to(torch.float32))
            preds = (output >= 0.5).float()

            val_epoch_loss += loss.item()
            val_epoch_acc += (preds == targets).float().mean()

            # f1-score
            val_precision.append(precision_score(targets.to('cpu'), preds.to('cpu')))
            val_recall.append(recall_score(targets.to('cpu'), preds.to('cpu')))
            val_f1.append(f1_score(targets.to('cpu'), preds.to('cpu')))
        
        current_f1 = np.mean(val_f1)
        val_epoch_loss /= len(val_loader)
        val_epoch_acc /= len(val_loader)
        
        print('====VAL_EPOCH_LOSS : {:.4f}===='.format(val_epoch_loss))
        print('====VAL_EPOCH_ACC : {:.4f}===='.format(val_epoch_acc))
        print('====VAL_PRECISION : {:.4f}===='.format(np.mean(val_precision)))
        print('====VAL_RECALL : {:.4f}===='.format(np.mean(val_recall)))
        print('====VAL_F1 : {:.4f}===='.format(np.mean(val_f1)))

        wandb.log({'VAL_EPOCH_LOSS' : val_epoch_loss})
        wandb.log({'VAL_EPOCH_ACC' : val_epoch_acc})    
        wandb.log({'VAL_PRECISION' : np.mean(val_precision)})
        wandb.log({'VAL_RECALL' : np.mean(val_recall)})
        wandb.log({'VAL_F1' : np.mean(val_f1)})
    
    if best_val_acc < val_epoch_acc:
        if epoch == 0:
            best_val_acc = val_epoch_acc
            torch.save(model.module.state_dict(), os.path.join(SAVE_PATH, f'{DATASET_NAME}_TL_{MODEL_NAME}_best_epoch_{epoch}.pth'))
            print("model saved!!")
        else:
            os.remove(os.path.join(SAVE_PATH, f'{DATASET_NAME}_TL_{MODEL_NAME}_best_epoch_{best_epoch}.pth'))
            best_epoch=epoch
            best_val_acc = val_epoch_acc
            torch.save(model.module.state_dict(), os.path.join(SAVE_PATH, f'{DATASET_NAME}_TL_{MODEL_NAME}_best_epoch_{best_epoch}.pth'))
            print("model_updated!!")

