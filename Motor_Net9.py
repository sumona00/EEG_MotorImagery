import torch.nn as nn
from torch.nn import init
import torch
from torch.autograd import Variable


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


class firstConvLayer(nn.Module):
    def __init__(self, input_tensor,
                 filters=50):
        super(firstConvLayer, self).__init__()


        self.conv1 = nn.Conv2d(input_tensor, filters,
                                   (3, 3),
                                   stride=1, padding = 'same' )
        init.xavier_uniform_(self.conv1.weight, gain=1)
        init.constant_(self.conv1.bias, 0)

        self.bnorm1 = nn.BatchNorm2d(filters,
                                    momentum=0.1,
                                    affine=True)
        init.constant_(self.bnorm1.weight, 1)
        init.constant_(self.bnorm1.bias, 0)


        self.lrelu1 =nn.LeakyReLU()

        # self.pool1 = nn.MaxPool2d(
        #     kernel_size=(3,3))
        
        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, data):
        x = self.lrelu1(self.bnorm1(self.conv1(data)))
        x = self.drop1(x)
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

        self.bnorm2 = nn.BatchNorm2d(filters,
                                    momentum=0.1,
                                    affine=True)
        init.constant_(self.bnorm2.weight, 1)
        init.constant_(self.bnorm2.bias, 0)


        self.lrelu1 =nn.LeakyReLU()
        self.lrelu2 =nn.LeakyReLU()

        # self.pool1 = nn.MaxPool2d(
        #     kernel_size=(3,3))

        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, data):
        x = self.conv1(data)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(self.bnorm2(x))
        x = self.drop1(x)
        
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

        self.bnorm2 = nn.BatchNorm2d(filters,
                                    momentum=0.1,
                                    affine=True)
        init.constant_(self.bnorm2.weight, 1)
        init.constant_(self.bnorm2.bias, 0)


        self.lrelu1 =nn.LeakyReLU()
        self.lrelu2 =nn.LeakyReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=(3,2))

        

    def forward(self, data):
        x = self.conv1(data)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(self.bnorm2(x))
        x = self.pool1(x)
        
        return x

class DpEEG_net(nn.Module):
    def __init__(self,
        out=3):
        super(DpEEG_net, self).__init__()

#         self.basenet = BaseNet(input_tensor=1,filters=[25,25])

        self.conv1 = firstConvLayer(input_tensor=1,filters=64)

        self.conv2 = ConvLayer(input_tensor=64,filters=64)

        # self.conv3 = ConvLayer(input_tensor=64,filters=128)

        self.conv3 = lastConvLayer(input_tensor=64,filters=128)

        self.conv4 = lastConvLayer(input_tensor=128, filters=128)

        self.lrelu = nn.LeakyReLU()
        # self.lrelu1 = nn.LeakyReLU()
        # self.lrelu3 = nn.LeakyReLU()

        self.pool1 = nn.AvgPool2d(
            kernel_size=(5,3))

        self.drop1 = nn.Dropout(p=0.5)

        self.drop2 = nn.Dropout(p=0.5)

        self.dense1 = nn.Linear(2048, out) # activation softmax 191488

        # self.dense2 = nn.Linear(50, out)

        self.lsoft = nn.LogSoftmax(dim=1)




    def forward(self, data):

        

        # data = data # (B, C, L, H) -> C = 22 and H = 1

        # data = data.squeeze(3) # (B, C, L) -> C = 22

        # data = data.permute(0,2,1) # (B, L, C) -> C = 22

        # print(lx.shape)

        # data, _ = self.lstm(data)

        # data = data.permute(0,2,1) # (B, L, C) -> C = 22

        # data = data[:,:,:,None]

        data = _transpose_time_to_spat(data)

        # print(x.shape)

        data = self.conv1(data)

        data = self.conv2(data)

        data = self.conv3(data)

        # data = self.conv4(data)

        data = self.pool1(self.conv4(data))

        # print(cx.shape)




        # x, (h, _) = self.lstm(lx)

        # print(data.shape)

        data = data.reshape(data.shape[0],-1)

        # # print(x.shape)

        data = self.lsoft(self.dense1(self.drop1(data)))

        # cx = self.lsoft(self.dense2(self.drop2(cx)))

        # print(x.shape)

        return data
