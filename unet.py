import torch
import torchvision
import lightning as L
from utils import *
from cust_dataset import *


class UnetEncoder(torch.nn.Module):
  def __init__(self, in_channel, num_filters):
    super().__init__()

    # layer convolutivo con 10 filtri 5x5
    self.conv1 = torch.nn.Conv2d(in_channel, num_filters, (3,3), padding='valid')
    self.batchnorm1 = torch.nn.BatchNorm2d(num_filters)
    self.conv2 = torch.nn.Conv2d(num_filters, num_filters, (3,3), padding='valid')
    self.batchnorm2 = torch.nn.BatchNorm2d(num_filters)
    self.pool = torch.nn.MaxPool2d(2, (2,2))


  def forward(self, x):
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = torch.nn.functional.relu(x)

    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = torch.nn.functional.relu(x)

    x = self.pool(x)

    return x



class UnetDecoder(torch.nn.Module):
  def __init__(self, in_channel, num_filters):
    super().__init__()


    self.tconv1 = torch.nn.ConvTranspose2d(in_channel, num_filters, (2,2), stride=2)




    self.conv1 = torch.nn.Conv2d(in_channel, num_filters, (3,3), padding='valid')
    self.batchnorm1 = torch.nn.BatchNorm2d(num_filters)
    self.conv2 = torch.nn.Conv2d(num_filters, num_filters, (3,3), padding='valid')
    self.batchnorm2 = torch.nn.BatchNorm2d(num_filters)



  def forward(self, x, skip_conn):

    x = self.tconv1(x)

    diffY = skip_conn.size()[2] - x.size()[2]
    diffX = skip_conn.size()[3] - x.size()[3]

    x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])


    x = torch.cat([x, skip_conn], dim=1)



    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = torch.nn.functional.relu(x)

    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = torch.nn.functional.relu(x)

    return x






class Unet(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.enc1 = UnetEncoder(3,64)
    self.enc2 = UnetEncoder(64,128)
    self.enc3 = UnetEncoder(128,256)
    self.enc4 = UnetEncoder(256,512)

    self.bot_conv1 = torch.nn.Conv2d(512,1024, (3,3), padding='valid')
    self.batchnorm1 = torch.nn.BatchNorm2d(1024)
    self.bot_conv2 = torch.nn.Conv2d(1024,1024, (3,3), padding='valid')
    self.batchnorm2 = torch.nn.BatchNorm2d(1024)


    self.dec1 = UnetDecoder(64*2, 64)
    self.dec2 = UnetDecoder(128*2, 128)
    self.dec3 = UnetDecoder(256*2, 256)
    self.dec4 = UnetDecoder(512*2, 512)

    self.conv_final = torch.nn.Conv2d(64, 1, (1,1), padding='valid')

  def forward(self, x):
    e1 = self.enc1.forward(x)
    e2 = self.enc2.forward(e1)
    e3 = self.enc3.forward(e2)
    e4 = self.enc4.forward(e3)


    x = self.bot_conv1(e4)
    x = self.batchnorm1(x)
    x = torch.nn.functional.relu(x)

    x = self.bot_conv2(x)
    x = self.batchnorm2(x)
    x = torch.nn.functional.relu(x)

    d4 = self.dec4.forward(x, e4)
    d3 = self.dec3.forward(d4, e3)
    d2 = self.dec2.forward(d3, e2)
    d1 = self.dec1.forward(d2, e1)

    x = self.conv_final(d1)

    return x




# define the LightningModule
class LitUnet(L.LightningModule):
    def __init__(self):
        self.example_input_array = torch.Tensor(1, 3, 500, 500) # per stampare le dimensioni
        super().__init__()
        self.net = Unet()


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_pred = self.net(x)
        y_pred = torchvision.transforms.functional.resize(y_pred, (y.shape[2], y.shape[3]))
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        loss = dice_loss(y_pred, y)
        y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()

        # Calcola l'accuratezza pixel-wise
        accuracy = (y_pred_bin == y).float().mean()
        f1_score = dice_coeff(y_pred_bin, y.int())

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("train_f1", f1_score, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
       # this is the test loop
        x, y = batch
        y_pred = self.net(x)
        y_pred = torchvision.transforms.functional.resize(y_pred, (y.shape[2], y.shape[3]))
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        loss = dice_loss(y_pred, y)
        y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()

        # Calcola l'accuratezza pixel-wise
        accuracy = (y_pred_bin == y).float().mean()
        f1_score = dice_coeff(y_pred_bin, y.int())

        # Log della loss e dell'accuratezza
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_pred = self.net(x)
        y_pred = torchvision.transforms.functional.resize(y_pred, (y.shape[2], y.shape[3]))
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y)
        loss = dice_loss(y_pred, y)

        y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()

        # Calcola l'accuratezza pixel-wise
        accuracy = (y_pred_bin == y).float().mean()
        f1_score = dice_coeff(y_pred_bin, y.int())

        # Log della loss e dell'accuratezza
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_f1", f1_score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


