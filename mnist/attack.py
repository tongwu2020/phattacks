from ROA import ROA
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision
from imshow import * 



parser = argparse.ArgumentParser()
parser.add_argument('--attlr',   type=float,   default=0.01, help='number of data loading workers')
parser.add_argument('--attiters', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--ROAwidth', type=int, default=5, help='The target class: 0 ')
parser.add_argument('--ROAheight', type=int, default=5, help='max number of iterations to find adversarial example')
parser.add_argument('--skip_in_x', type=int, default=1, help='Number of training images')
parser.add_argument('--skip_in_y', type=int, default=1, help='Number of test images')
parser.add_argument('--batch_size', type=int, default=128, help='1 == plot all successful adversarial images')
parser.add_argument('--restart', type=int, default=10, help='1 == plot all successful adversarial images')
parser.add_argument('--attackmodel', type=str, help='1 == plot all successful adversarial images')
parser.add_argument('--potential_nums', type=int, default=50, help='the height / width of the input image to network')
opt = parser.parse_args()
print(opt)



# loading data 
train_dataset= datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset= datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
batch_size = opt.batch_size
train_load=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_load=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("Number of images in training set: {}".format(len(train_dataset)))
print("Number of images in test set: {}".format(len(test_dataset)))
print("Number of batches in the train loader: {}".format(len(train_load)))
print("Number of batches in the test loader: {}".format(len(test_load)))



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
model.load_state_dict(torch.load('./model/'+ opt.attackmodel + '.pt'))
model.eval()

correct = 0 
total = 0 
torch.manual_seed(12345)
for i, (images,labels) in enumerate(test_load):
    
    images =images.cuda()
    labels =labels.cuda()
    roa = ROA(model, 28) 
    learning_rate = opt.attlr
    iterations = opt.attiters
    ROAwidth = opt.ROAwidth
    ROAheight = opt.ROAheight
    skip_in_x = opt.skip_in_x
    skip_in_y = opt.skip_in_y
    potential_nums = opt.potential_nums

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    check_num = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device)
    correct_num = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device) + opt.restart
    for i in range(opt.restart):
        roaimages = roa.gradient_based_search(images, labels, learning_rate,\
                        iterations, ROAwidth , ROAheight, skip_in_x, skip_in_y, potential_nums, True)
        imshow("testattack",roaimages.data)
        outputs = model(roaimages)
        _, predicted = torch.max(outputs.data, 1)
        check_num += (predicted == labels)
        #print(check_num)
    correct += (correct_num == check_num).sum().item()
    #print(correct)
    total += labels.size(0)
print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
print(opt)





