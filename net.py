import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as optim

class MyDataset(torch.utils.data.Dataset):  # 创类：MyDataset,继承torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 打开txt，读取内容
        imgs = []
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除本行string字符串末尾的指定字符
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，words[0]是图片信息，words[1]是label

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path
        img = Image.open(fn).convert('RGB')  # from PIL import Image

        if self.transform is not None:  # 是否进行transform
            img = self.transform(img)
        return img, label  # return回哪些内容，在训练时循环读取每个batch，就能获得哪些内容

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.imgs)


BATCH_SIZE = 20
LEARNING_RATE = 0.01
EPOCH = 20
N_CLASSES = 10

train_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

train_range_data = MyDataset(datatxt='train_range.txt', transform=train_transforms)
test_range_data = MyDataset(datatxt='val_range.txt', transform=train_transforms)

train_range_loader = torch.utils.data.DataLoader(dataset=train_range_data, batch_size=BATCH_SIZE, shuffle=True)
test_range_loader = torch.utils.data.DataLoader(dataset=train_range_data, batch_size=BATCH_SIZE, shuffle=False)


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

      
vgg16 = VGG16(n_classes=N_CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = vgg16.to(device)

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()#损失函数设置为交叉熵损失
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# epoch = 10
# for epoch in range(epoch):
#     sum_loss = 0.0
#     train_acc = 0
#     net.train()  #训练模式
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         # 前向传播
#         optimizer.zero_grad()  #将梯度归零
#         out = net(inputs)  #将数据传入网络进行前向运算
#         loss = criterion(out, labels)  #得到损失函数
#         # 反向传播
#         #optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # 记录误差
#         sum_loss += loss.item()
#         if i % 100 == 99:
#             print('[%d,%d/%d({%.3f})] loss:%.03f' % (epoch + 1,64. * (i+1) , len(train_loader.dataset),100. * (i+1) / len(train_loader), sum_loss / 100))

#             sum_loss = 0.0
# Train the model
for epoch in range(EPOCH):

    sum_loss = 0
    #cnt = 0
    net.train() #训练模式
    for i,(images, labels) in enumerate(train_range_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward + Backward + Optimize
        optimizer.zero_grad()#将梯度归零
        _, outputs = net(images)#将数据传入网络进行前向运算
        loss = cost(outputs, labels)#得到损失函数
        #avg_loss += loss.data
        #cnt += 1
        #print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
        loss.backward()
        optimizer.step()
    #scheduler.step(avg_loss)
        sum_loss += loss.item()
        if i % 10 == 9:
            print('[%d,%d/%d({%.3f})] loss:%.03f' % (epoch + 1,20. * (i+1) , len(train_range_loader.dataset),100. * (i+1) / len(train_range_loader), sum_loss / 10))

            sum_loss = 0.0

# Test the model
net.eval()
correct = 0
total = 0

for i,(images, labels) in enumerate(test_range_loader):
    images = images.to(device)
    labels = labels.to(device)
    _, outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100* correct/total))

# Save the Trained Model
#torch.save(vgg16.state_dict(), 'cnn.pkl')