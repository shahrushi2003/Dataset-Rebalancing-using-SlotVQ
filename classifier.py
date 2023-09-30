import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import lightning
from lightning.pytorch import seed_everything

class FinalClassifier(nn.Module):
  def __init__(self, input_size, hid_num, output_size):
    super().__init__()
    self.input_size = input_size
    self.hid_num = hid_num
    self.output_size = output_size
    self.mlp1 = nn.Linear(input_size, hid_num)
    self.mlp2 = nn.Linear(hid_num, output_size)

  def forward(self, x):
    x = x.view(-1, self.input_size)
    x = self.mlp1(x)
    x = F.relu(x)
    x = self.mlp2(x)
    return x

class Final_Conv_Classifier(nn.Module):
  def __init__(self, input_size, in_channels, output_size):
    super().__init__()
    self.input_size = input_size
    self.in_channels = in_channels
    self.output_size = output_size
    self.conv1 = nn.Conv2d(self.in_channels, 16, 3)
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.pool2 = nn.MaxPool2d(2)
    self.conv3 = nn.Conv2d(32, output_size, 3)
    self.gap = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.gap(x)
    x = x.view(x.shape[0], -1)
    return x


class Lightning_Classifier(lightning.LightningModule):
    def __init__(self, opt, data_type):
        super().__init__()
        seed_everything(seed=opt["seed"], workers=True)
        self.data_type = data_type
        self.example_input_array = torch.Tensor(32, 3, 28, 28)
        self.opt = opt
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.opt["model_type"] == 'Conv':
          self.model = Final_Conv_Classifier(28, 3, 10)
        elif self.opt["model_type"] == 'MLP':
          self.model = FinalClassifier(28*28*3, 10, 10)
        self.metric = Accuracy('multiclass', num_classes=opt["num_classes"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, _, labels = batch
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log(f"{self.data_type}_train_loss", loss, batch_size=self.opt['batch_size'])
        acc = self.metric(outputs, labels)
        self.log(f"{self.data_type}_train_accuracy", acc, batch_size=self.opt['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        images, _, labels = batch
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log(f"{self.data_type}_val_loss", loss, batch_size=self.opt['batch_size'])
        acc = self.metric(outputs, labels)
        acc_name = f"{self.data_type}_test_accuracy" if dataloader_idx else  f"{self.data_type}_val_accuracy"
        self.log(acc_name, acc, batch_size=self.opt['batch_size'])
        return loss

    def test_step(self, batch, batch_idx):
        images, _, labels = batch
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log(f"{self.data_type}_test_loss", loss, batch_size=self.opt['batch_size'])
        acc = self.metric(outputs, labels)
        self.log(f"{self.data_type}_test_accuracy", acc, batch_size=self.opt['batch_size'])
        return loss

    def predict_step(self, batch, batch_idx):
        images, _, labels = batch
        outputs = self.model(images)
        return outputs

    def configure_optimizers(self):
        params = [{'params': self.model.parameters()}]
        optimizer = optim.Adam(params, lr = self.opt["learning_rate"])
        return optimizer
