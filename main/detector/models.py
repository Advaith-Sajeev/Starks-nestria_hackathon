from torchvision import transforms
from torch.utils.data.dataset import Dataset
import cv2
import torch
from torch import nn
from torchvision import models
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection


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
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add a singleton dimension for seq_length
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def predict(model, img):
    with torch.no_grad():
        fmap, logits = model(img.to('cuda'))
    probs = nn.functional.softmax(logits, dim=1)
    confidence = probs[:, 1].item() * 100  # Assuming 1 is the index for the positive class
    return confidence


class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []

        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

        for i, frame in enumerate(self.frame_extract(video_path)):
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(
                        bboxC.height * ih)
                    face = frame[y:y + h, x:x + w]
                    frames.append(self.transform(face))

                    if len(frames) == self.count:
                        break

        if not frames:
            return torch.zeros((1, 3, im_size, im_size))

        frames = torch.stack(frames)[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


def process_and_predict(path_to_videos, model_name):
    path_to_videos = [path_to_videos]
    video_dataset = ValidationDataset(path_to_videos, sequence_length=20, transform=train_transforms)
    model = Model(2).cuda()
    path_to_model = rf"C:\Users\mahes\ML\NESTRIS\main\Models\{model_name}.pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    print(path_to_videos[0])
    prediction = predict(model, video_dataset[0])
    return prediction

# Example usage:
# process_and_predict("path_to_your_video.mp4", "your_model_name")
