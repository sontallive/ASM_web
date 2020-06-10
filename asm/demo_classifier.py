from asm.net.resnet_classifier import rsnet50
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import cv2
import os

model_classifier_A = None
model_classifier_a = None

scale = (224, 224)
resize_img = transforms.Resize(scale, Image.BILINEAR)
to_tensor = transforms.ToTensor()
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def load_classification_model():
    global model_classifier_A
    global model_classifier_a

    # model
    print('load classification model...')
    model_classifier_A = rsnet50(pretrained=False, num_class=10)
    model_classifier_a = rsnet50(pretrained=False, num_class=10)
    ckpt_classifier_A = os.path.join(os.path.dirname(__file__), 'ckpt/cigar_bar_200.pth')
    ckpt_classifier_a = os.path.join(os.path.dirname(__file__), 'ckpt/cigar_box_180.pth')
    print('load {}'.format(ckpt_classifier_A))
    print('load {}'.format(ckpt_classifier_a))

    if torch.cuda.is_available():
        model_classifier_A = torch.nn.DataParallel(model_classifier_A).cuda()
        model_classifier_A.module.load_state_dict(torch.load(ckpt_classifier_A))

        model_classifier_a = torch.nn.DataParallel(model_classifier_a).cuda()
        model_classifier_a.module.load_state_dict(torch.load(ckpt_classifier_a))
    else:
        model_classifier_A.load_state_dict(torch.load(model_classifier_A, map_location='cpu'))
        model_classifier_a.load_state_dict(torch.load(model_classifier_a, map_location='cpu'))
    print('done!')
    model_classifier_A.eval()
    model_classifier_a.eval()


@torch.no_grad()
def classify_A(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = resize_img(img)
    img = to_tensor(img).float()
    img = normalize_img(img)
    img = img.unsqueeze(0)
    res = model_classifier_A(img.cuda()).squeeze()
    res = reverse_one_hot(res)
    return int(res)


@torch.no_grad()
def classify_a(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = resize_img(img)
    img = to_tensor(img).float()
    img = normalize_img(img)
    img = img.unsqueeze(0)
    res = model_classifier_a(img.cuda()).squeeze()
    res = reverse_one_hot(res)
    return int(res)


def reverse_one_hot(x):
    x = torch.argmax(x, dim=-1)
    return x.cpu().numpy()
