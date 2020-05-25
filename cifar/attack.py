import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
from imshow import *
from ROA import ROA
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=DATASETS) #cifar10
parser.add_argument('arch', type=str, choices=ARCHITECTURES) #cifar_resnet110
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--attlr',   type=float,   default=0.05, help='number of data loading workers')
parser.add_argument('--batch', default=512, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--attiters', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--ROAwidth', type=int, default=5, help='The target class: 0 ')
parser.add_argument('--ROAheight', type=int, default=5, help='max number of iterations to find adversarial example')
parser.add_argument('--skip_in_x', type=int, default=1, help='Number of training images')
parser.add_argument('--skip_in_y', type=int, default=1, help='Number of test images')
parser.add_argument('--potential_nums', type=int, default=50, help='the height / width of the input image to network')
parser.add_argument('--base_classifier', type=str, default="ori", help='1 == plot all successful adversarial images')
args = parser.parse_args()
print(args)


checkpoint = torch.load(args.base_classifier)
model = get_architecture(checkpoint["arch"], args.dataset)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
criterion = CrossEntropyLoss().cuda()

test_dataset = get_dataset(args.dataset, 'test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=4, pin_memory=False)

logfilename = os.path.join(args.outdir, 'log.txt')
log(logfilename, "{0}".format(args))



correct = 0 
total = 0 
torch.manual_seed(12345)
for i, (images,labels) in enumerate(test_loader):
    
    images =images.cuda()
    labels =labels.cuda()
    roa = ROA(model, 32) 

    learning_rate = args.attlr
    iterations = args.attiters
    ROAwidth = args.ROAwidth
    ROAheight = args.ROAheight
    skip_in_x = args.skip_in_x
    skip_in_y = args.skip_in_y
    potential_nums = args.potential_nums
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    check_num = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device)
    correct_num = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device) + 10
    for i in range(10):
        roaimages = roa.gradient_based_search(images, labels, learning_rate,\
                        iterations, ROAwidth , ROAheight, skip_in_x, skip_in_y, potential_nums, True)
        imshow("testattack",roaimages.data)
        outputs = model(roaimages)
        _, predicted = torch.max(outputs.data, 1)
        #print(type(predicted),type(labels),type(check_num))
        check_num += (predicted == labels).byte()
        #print(check_num)
    correct += (correct_num == check_num).sum().item()
    #print(correct)
    total += labels.size(0)
log(logfilename,'Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))

