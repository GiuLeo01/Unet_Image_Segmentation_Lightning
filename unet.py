

class UnetEncoder(torch.nn.Module):
  def __init__(self, in_channel, num_filters):
    super().__init__()

    # layer convolutivo con 10 filtri 5x5
    self.conv1 = torch.nn.Conv2d(in_channel, num_filters, (3,3))
    self.conv2 = torch.nn.Conv2d(num_filters, num_filters, (3,3))
    self.pool = torch.nn.MaxPool2d(2, 2)


  def forward(self, x):
    x = self.conv1(x)
    x = torch.nn.functional.relu(x)

    x = self.conv2(x)
    x = torch.nn.functional.relu(x)
    
    x = self.pool(x)

    return x
  


class UnetDecoder(torch.nn.Module):
  def __init__(self, in_channel, num_filters):
    super().__init__()

  
    self.tconv1 = torch.nn.ConvTranspose2d(in_channel, num_filters, (2,2), stride=2)



    
    self.conv1 = torch.nn.Conv2d(in_channel, num_filters, (3,3))
    self.conv2 = torch.nn.Conv2d(num_filters, num_filters, (3,3))



  def forward(self, x, skip_conn):

    print(x.shape)
    x = self.tconv1(x)

    skip_conn = torchvision.transforms.functional.resize(skip_conn, (x.shape[2], x.shape[3]))

    print(skip_conn.shape, x.shape)

    x = torch.cat([x, skip_conn], dim=1)

    print(x.shape)


    x = self.conv1(x)
    x = torch.nn.functional.relu(x)

    x = self.conv2(x)
    x = torch.nn.functional.relu(x)

    return x
  





class Unet(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.enc1 = UnetEncoder(3,64)
    self.enc2 = UnetEncoder(64,128)
    self.enc3 = UnetEncoder(128,256)
    self.enc4 = UnetEncoder(256,512)
    
    self.bot_conv1 = torch.nn.Conv2d(512,1024, (3,3))
    self.bot_conv2 = torch.nn.Conv2d(1024,1024, (3,3))

    
    self.dec1 = UnetDecoder(64*2, 64)
    self.dec2 = UnetDecoder(128*2, 128)
    self.dec3 = UnetDecoder(256*2, 256)
    self.dec4 = UnetDecoder(512*2, 512)

    self.conv_final = torch.nn.Conv2d(64, 1, (1,1))
  
  def forward(self, x):
    e1 = self.enc1.forward(x)
    e2 = self.enc2.forward(e1)
    e3 = self.enc3.forward(e2)
    e4 = self.enc4.forward(e3)


    x = self.bot_conv1(e4)
    x = torch.nn.functional.relu(x)

    x = self.bot_conv2(x)
    x = torch.nn.functional.relu(x)

    d4 = self.dec4.forward(x, e4)
    d3 = self.dec3.forward(d4, e3)
    d2 = self.dec2.forward(d3, e2)
    d1 = self.dec1.forward(d2, e1)

    x = torch.nn.functional.sigmoid(self.conv_final(d1))

    return x


