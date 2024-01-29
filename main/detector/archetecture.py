import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(10, 1024)
        self.relu1 = nn.LeakyReLU(0.2)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.LeakyReLU(0.2)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.LeakyReLU(0.2)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 128)
        self.relu4 = nn.LeakyReLU(0.2)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.LeakyReLU(0.2)
        self.batch_norm5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(64, 32)
        self.relu6 = nn.LeakyReLU(0.2)
        self.batch_norm6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(0.1)

        self.fc7 = nn.Linear(32, 16)
        self.relu7 = nn.LeakyReLU(0.2)
        self.batch_norm7 = nn.BatchNorm1d(16)
        self.dropout7 = nn.Dropout(0.1)

        self.fc_final = nn.Linear(16, 1)  # Adjusted output size to match the final layer in your TensorFlow model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.batch_norm1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.batch_norm2(self.relu2(self.fc2(x))))
        x = self.dropout3(self.batch_norm3(self.relu3(self.fc3(x))))
        x = self.dropout4(self.batch_norm4(self.relu4(self.fc4(x))))
        x = self.dropout5(self.batch_norm5(self.relu5(self.fc5(x))))
        x = self.dropout6(self.batch_norm6(self.relu6(self.fc6(x))))
        x = self.dropout7(self.batch_norm7(self.relu7(self.fc7(x))))
        x = self.sigmoid(self.fc_final(x))
        return x