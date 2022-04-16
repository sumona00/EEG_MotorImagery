import torch.nn as nn
from torch.nn import init
import torch
from torch.autograd import Variable

class firstConvLayer(nn.Module):
    def __init__(self, input_tensor,
                 filters=50):
        super(firstConvLayer, self).__init__()


        self.bnorm1 = nn.BatchNorm2d(input_tensor,
                                    momentum=0.1,
                                    affine=True)
        init.constant_(self.bnorm1.weight, 1)
        init.constant_(self.bnorm1.bias, 0)

        self.conv1 = nn.Conv2d(input_tensor, filters,
                                   (3, 3),
                                   stride=1, padding = 'same' )
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)


        self.lrelu1 =nn.LeakyReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=(3,2))
        
        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, data):
        x = self.bnorm1(data)
        x = self.lrelu1(self.conv1(x))
        x = self.drop1(self.pool1(x))
        return x



class ConvLayer(nn.Module):
    def __init__(self, input_tensor,
                 filters=50):
        super(ConvLayer, self).__init__()

        

        self.conv1 = nn.Conv2d(input_tensor, filters,
                                   (3, 3),
                                   stride=1, padding = 'same' )
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(filters, filters,
                                   (3, 3),
                                   stride=1, padding = 'same' )
        init.xavier_uniform_(self.conv2.weight, gain=1)
        init.constant_(self.conv2.bias, 0)


        self.lrelu1 =nn.LeakyReLU()
        self.lrelu2 =nn.LeakyReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=(3,2))

        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, data):
        x = self.conv1(data)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.drop1(self.pool1(x))
        
        return x


class lastConvLayer(nn.Module):
    def __init__(self, input_tensor,
                 filters=50):
        super(lastConvLayer, self).__init__()

        

        self.conv1 = nn.Conv2d(input_tensor, filters,
                                   (3, 3),
                                   stride=1, padding = 'same' )
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(filters, filters,
                                   (3, 3),
                                   stride=1, padding = 'same' )
        init.xavier_uniform_(self.conv2.weight, gain=1)
        init.constant_(self.conv2.bias, 0)


        self.lrelu1 =nn.LeakyReLU()
        self.lrelu2 =nn.LeakyReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=(3,2))

        

    def forward(self, data):
        x = self.conv1(data)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.pool1(x)
        
        return x

class DpEEG_net(nn.Module):
    def __init__(self,
        out=3):
        super(DpEEG_net, self).__init__()

#         self.basenet = BaseNet(input_tensor=1,filters=[25,25])

        self.conv1 = firstConvLayer(input_tensor=2,filters=64)

        self.conv2 = ConvLayer(input_tensor=64,filters=64)

        self.conv3 = ConvLayer(input_tensor=64,filters=128)

        self.conv4 = lastConvLayer(input_tensor=128,filters=128)

        # self.conv4 = lastConvLayer(input_tensor=128, filters=128)

        self.lstm = nn.LSTM(22, 2, 1, batch_first=True, bidirectional=True)

        self.lrelu = nn.LeakyReLU()
        # self.lrelu1 = nn.LeakyReLU()
        # self.lrelu3 = nn.LeakyReLU()

        self.drop1 = nn.Dropout(p=0.5)

        self.drop2 = nn.Dropout(p=0.5)

        self.dense1 = nn.Linear(3000, 50) # activation softmax 191488

        self.dense2 = nn.Linear(50, out)

        self.lsoft = nn.LogSoftmax(dim=1)




    def forward(self, data):

        

        data_lstm = data # (B, C, L, H) -> C = 22 and H = 1

        lx = data_lstm.squeeze(3) # (B, C, L) -> C = 22

        lx = lx.permute(0,2,1) # (B, L, C) -> C = 22

        # print(lx.shape)

        # data = _transpose_time_to_spat(data)

        # cx = self.conv1(data)

        # cx = self.conv2(cx)

        # cx = self.conv3(cx)

        # cx = self.conv4(cx)

        # print(x.shape)




        x, (h, _) = self.lstm(lx)

        # print(x.shape)

        x = x.reshape(x.shape[0],-1)

        # print(x.shape)

        x = self.lrelu(self.dense1(self.drop1(x)))

        x = self.lsoft(self.dense2(self.drop2(x)))

        # print(x.shape)

        return x
