import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='1 == plot all successful adversarial images')
opt = parser.parse_args()
print(opt)




test_dataset= datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
batch_size = 1000
test_load=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



#build CNN classifier 
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # input_size:28, same_padding=(filter_size-1)/2, 3-1/2=1:padding
        self.cnn1=nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # input_size-filter_size +2(padding)/stride + 1 = 28-3+2(1)/1+1=28
        self.batchnorm1=nn.BatchNorm2d(8)
        # output_channel:8, batch(8)
        self.relu=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        #input_size=28/2=14
        self.cnn2=nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        # same_padding: (5-1)/2=2:padding_size. 
        self.batchnorm2=nn.BatchNorm2d(32)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        # input_size=14/2=7
        # 32x7x7=1568
        self.fc1 =nn.Linear(in_features=1568, out_features=600)
        self.dropout= nn.Dropout(p=0.5)
        self.fc2 =nn.Linear(in_features=600, out_features=10)
    def forward(self,x):
        out =self.cnn1(x)
        out =self.batchnorm1(out)
        out =self.relu(out)
        out =self.maxpool1(out)
        out =self.cnn2(out)
        out =self.batchnorm2(out)
        out =self.relu(out)
        out =self.maxpool2(out)
        out =out.view(-1,1568)
        out =self.fc1(out)
        out =self.relu(out)
        out =self.dropout(out)
        out =self.fc2(out)
        return out


model=CNN()
model.load_state_dict(torch.load('./model/'+ opt.model + '.pt'))
model.eval()
model.cuda()

correct =0
total =0
for images,labels in test_load:
    images =Variable(images.cuda())
    outputs=model(images)
    _,predicted=torch.max(outputs.data,1)
    total+=labels.size(0)
    correct += (predicted.cpu()==labels.cpu()).sum()
    print(correct)
accuracy = correct.cpu().numpy()/total
print(accuracy)
