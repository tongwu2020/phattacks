# The code is based on https://colab.research.google.com/github/omerbsezer/Fast-Pytorch/blob/master/Learning_Pytorch/CNN_Mnist.ipynb#scrollTo=3ve5cf1rBYJ-
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision
#torch.cuda.is_available()
#import matplotlib.pyplot as plt
import argparse
from ROA import ROA


parser = argparse.ArgumentParser()
parser.add_argument('--attlr',   type=float,   default=0.05, help='number of data loading workers')
parser.add_argument('--attiters', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--ROAwidth', type=int, default=5, help='The target class: 0 ')
parser.add_argument('--ROAheight', type=int, default=5, help='max number of iterations to find adversarial example')
parser.add_argument('--skip_in_x', type=int, default=1, help='Number of training images')
parser.add_argument('--skip_in_y', type=int, default=1, help='Number of test images')
parser.add_argument('--potential_nums', type=int, default=20, help='the height / width of the input image to network')
parser.add_argument('--batch_size', type=int, default=128, help='1 == plot all successful adversarial images')
parser.add_argument('--epochs', type=int, default=30, help='1 == plot all successful adversarial images')
opt = parser.parse_args()
print(opt)




# loading data 
train_dataset= datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset= datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
batch_size = opt.epochs
epochs= opt.epochs
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

# Set up the model
model=CNN()
CUDA=torch.cuda.is_available()
if CUDA:
    model=model.cuda()
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.01)

iteration=0
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_load):
        iteration+=1
        if CUDA:
            images =Variable(images.cuda())
            labels =Variable(labels.cuda())
        else:
            images =Variable(images)
            labels =Variable(labels)
        #initialize the ROA module
        #Training parm
        roa = ROA(model, 28) 
        learning_rate = opt.attlr
        iterations = opt.attiters
        ROAwidth = opt.ROAwidth
        ROAheight = opt.ROAheight
        skip_in_x = opt.skip_in_x
        skip_in_y = opt.skip_in_y
        potential_nums = opt.potential_nums


        # Image = roa.gradient_based_search(images, labels, learning_rate,\
        #      iterations, ROAwidth , ROAheight, skip_in_x, skip_in_y, potential_nums)

        optimizer.zero_grad()
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        test_roa = ROA(model,28)
        
        if(i+1)%100 ==0:
            correct =0
            correct_adv = 0 
            total =0
            for images,labels in test_load:
                if CUDA:
                    images =Variable(images.cuda())
                else:
                    images =Variable(images)
                #Image = roa.gradient_based_search(images, labels, Attack_learning_rate,\
                #   Attack_iterations, ROAwidth , ROAheight, skip_in_x, skip_in_y, attack_pot_nums)
                Image = test_roa.gradient_based_search(images, labels, learning_rate,\
                        iterations, ROAwidth , ROAheight, skip_in_x, skip_in_y, potential_nums)
                outputs_adv=model(Image)
                _,predicted_adv=torch.max(outputs_adv.data,1)

                outputs=model(images)
                _,predicted=torch.max(outputs.data,1)

                total+=labels.size(0)
                if CUDA:
                    correct += (predicted.cpu()==labels.cpu()).sum()
                    correct_adv += (predicted_adv.cpu()==labels.cpu()).sum()
                else:
                    correct += (predicted==labels).sum()
                    correct_adv += (predicted_adv==labels).sum()

            accuracy = 100 *correct/total
            accuracy_adv = 100 *correct_adv/total
            print("Iteration: {}, Train Loss: {}, Test Accuracy:{}%, Adv_test Accuracy:{}%".format(iteration, loss.item(),accuracy, accuracy_adv))


torch.save(model.state_dict(), './model/doa1.pt')  
print("Finished!")
print(" learning_rate " , learning_rate ,
      " iterations    " , iterations,
      " ROAwidth      " , ROAwidth,
      " ROAheight     " , ROAheight,
      " skip_in_x     " , skip_in_x,
      " skip_in_y     " , skip_in_y,
      " potential_nums" , potential_nums,
      " epochs        " , epochs,
      " batch_size    " , batch_size
        )
### We can find that the ROA retrain do not downgrade the clean accuracy (same proformance) and we have 85 % of accuracy of ROA :)
### You can play around with it, find out how to train a more robust model:)
