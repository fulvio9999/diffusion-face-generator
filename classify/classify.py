#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# last mod 24/05/2023 17.48
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm
from PIL import Image as PILImage
try:
    from facenet_pytorch import InceptionResnetV1
except:
    print('Missing lib. Install with')
    print('pip install facenet-pytorch')
    sys.exit(1)


CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print('DEVICE', device)


class RealClassifierModel(nn.Module):
    def __init__(self, input_dim=3*64*64):
        super().__init__()
        
        self.features = InceptionResnetV1(pretrained='vggface2').eval()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
            


def group_images(fname_iter):
    groups = defaultdict(list)
    
    for fname in fname_iter:
        i = fname.name.index('_')
        
        image_name = fname.name[:i]
        groups[image_name].append(fname)
    
    assert len(groups) == 50, 'missing images'
    assert sum([1 if len(x) != 10 else 0 for x in groups.values()]) == 0, 'missing samples'
    
    return groups


def load_images(image_list):
    
    T = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    x_list = [T(PILImage.open(str(x))).unsqueeze(0) for x in image_list]
    
    return torch.vstack(x_list)
    

def main(folder_name):
    
    model_path = Path(__file__).parent / 'model_0605.pth'
    
    test_set_directives = group_images(Path(folder_name).glob('*.jpg'))
    print(f'{len(test_set_directives)=}')
    
    model = RealClassifierModel()
    print('LOAD', model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_outputs_prob = []

    with torch.no_grad():

        for image_list in tqdm(test_set_directives.values()):

            inputs = load_images(image_list)

            # Generate outputs
            outputs = model(inputs.to(device))
            all_outputs_prob.append(outputs.detach().cpu().max().item())
            
    print('\n\n\n')
    print(f'Classification: {np.mean(all_outputs_prob)}')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print(f'Usage: {sys.argv[0]} path/to/folder')
