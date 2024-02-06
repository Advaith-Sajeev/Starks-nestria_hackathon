import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from torch import nn
import os
import numpy as np
from torchvision import models
import mediapipe as mp


# Model definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# Other helper functions and constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))


# Function to convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


# Function to generate image
def generate_image(fmap, logits, weight_softmax, img):
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :])
    result = heatmap * 0.5 + img * 0.8 * 255
    result1 = heatmap * 0.5 / 255 + img * 0.8

    # Combine heatmap and original image
    result_rgb = cv2.cvtColor(result1.astype(np.uint8), cv2.COLOR_BGR2RGB)

    return result_rgb


# Function to process video using Mediapipe for face detection
def process_video(video_path, sequence_length=20, transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])):
    frames = []
    a = int(100 / sequence_length)
    first_frame = np.random.randint(0, a)

    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    for i, frame in enumerate(frame_extract(video_path)):
        # Convert frame to RGB (Mediapipe uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection using Mediapipe
        result = face_detection.process(frame_rgb)

        # If a face is detected, crop the frame to the face region
        if result.detections:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                frame = frame[y:y + h, x:x + w]

        frames.append(transform(frame))
        if len(frames) == sequence_length:
            break

    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames.unsqueeze(0)


# Function to make predictions
def predict(model, video_object):
    fmap, logits = model(video_object.to('cuda'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return fmap, logits, weight_softmax, confidence


# Function to extract frames from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

#
# # Example usage:
# video_path = r"C:\Users\mahes\ML\dataset\data unziped\dfdc_train_part_0\xpzfhhwkwb.mp4"
#
# video_object = process_video(video_path, sequence_length=20)
#
# model = Model(2).cuda()
# path_to_model = '../Models/model_84_acc_10_frames_final_data.pt'
# model.load_state_dict(torch.load(path_to_model))
# model.eval()
#
# fmap, logits, weight_softmax , con = predict(model, video_object)
# print(con)
# # Example usage of generate_image
# generate_image(fmap, logits, weight_softmax, video_object, path='./')
