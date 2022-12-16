
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch import optim
from model.mesonet.network.classifier import Meso4, MesoInception4

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, auc

import argparse
import random
import numpy as np
import os

from tqdm import tqdm

import PIL.Image as Image

# import wandb


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
parser.add_argument('--gpus', default='0', type=str, help='id of gpus to use')
parser.add_argument('--num_gpus', default=1, type=int, help='numbers of gpus to use')
parser.add_argument('--ckpt', type=str, help='file name of model weight')
args = parser.parse_args()


# TEST_PATH = '/media/data1/donggeun/masked_crop_test/FakeAVCeleb'
TEST_PATH = '/home/data/deepfake_privacy/FF_original/FaceForensics++/FaceSwap/test/'
# TEST_PATH = '/home/data/deepfake_privacy/ff_priv/NeuralTextures/pixel/test/'
SAVE_PATH = "/home/donggeun/ckpt/DeepFake"
MODEL_NAME = 'mesoinception4'

ckpt_path = '/home/donggeun/kaia2022/checkpoint/FaceSwap_mesoinception4_best_epoch_4.pth'
# MODEL_NAME = 'efficientnet_b0'
#MODEL_NAME = 'mobilenetv2_100'
# MODEL_NAME = 'meso4'
# MODEL_NAME = 'meso_inception4'
DATASET_NAME = TEST_PATH.split("/")[-1]


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus  

# wandb.init(project=DATASET_NAME + '_' + MODEL_NAME, entity="dlwns147")


# wandb.config = {
#     "learning_rate": lr,
#     "epochs": num_epochs,
#     "batch_size": args.batch_size,
# }
# wandb.run.name = args.dataname

transform = transforms.Compose([
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

test_dataset = datasets.ImageFolder(TEST_PATH, transform=transform)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = 'entire_model.pth'
# model = torch.load(model_path).to(device)

if MODEL_NAME == 'mesoinception4':
    model = MesoInception4(num_classes=1).to(device)

else:
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1).to(device)

# model = Meso4(num_classes=1).to(device)
# model = MesoInception4(num_classes=1).to(device)

# model.load_state_dict(torch.load(os.path.join(SAVE_PATH, args.ckpt)))
model.load_state_dict(torch.load(ckpt_path))
model = nn.DataParallel(model, device_ids=range(args.num_gpus))

best_test_f1 = 0
best_epoch = 0
best_test_acc = 0

test_acc = 0
test_precision = []
test_recall = []
test_f1 = []
test_auc = []
test_real_auc = []
test_epoch_acc = 0

model.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(tqdm(test_loader)):
        inputs, targets = data

        # targets = targets.unsqueeze(1).type(torch.FloatTensor)
        inputs, targets = inputs.to(device), targets.to(device)

        output = model(inputs).squeeze()
        preds = (output >= 0.5).float()

        test_epoch_acc += (preds == targets).float().mean()

        # f1-score
        test_precision.append(precision_score(targets.to('cpu'), preds.to('cpu')))
        test_recall.append(recall_score(targets.to('cpu'), preds.to('cpu')))
        test_f1.append(f1_score(targets.to('cpu'), preds.to('cpu')))
        test_auc.append(roc_auc_score(targets.cpu(), preds.cpu()))
        # test_real_auc.append(auc(targets.cpu(), preds.cpu()))


    
    current_f1 = np.mean(test_f1)
    test_epoch_acc /= len(test_loader)
    
    print('====TEST_EPOCH_ACC : {:.4f}===='.format(test_epoch_acc))
    print('====TEST_PRECISION : {:.4f}===='.format(np.mean(test_precision)))
    print('====TEST_RECALL : {:.4f}===='.format(np.mean(test_recall)))
    print('====TEST_F1 : {:.4f}===='.format(np.mean(test_f1)))
    print('====TEST_ROC_AUC : {:.4f}===='.format(np.mean(test_auc)))
    # print('====TEST_AUC_SCORE : {:.4f}===='.format(np.mean(test_real_auc)))


    # wandb.log({'TEST_EPOCH_ACC' : test_epoch_acc})    
    # wandb.log({'TEST_PRECISION' : np.mean(test_precision)})
    # wandb.log({'TEST_RECALL' : np.mean(test_recall)})
    # wandb.log({'TEST_F1' : np.mean(test_f1)})


