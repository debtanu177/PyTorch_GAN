import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys


# Initializing required parameters
batch_size = 32
max_epoch = 500
learning_rate = 2e-4
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
num_workers = 10
load_epoch = -1
latent_dim = 100

def load_data():
    # this function loads MNIST dataset and returns train and test loader
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader


class Generator(nn.Module):
    # implementation of the generator network
    def __init__(self, input_dim=20, output_dim=784):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 784)
        
        self.activation = nn.ReLU()
#         self.activation1 = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        return x


class Discriminator(nn.Module):
    # implementation of the discriminator module
    def __init__(self, input_dim=784, output_dim=1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_dim)
        
        self.activation = nn.ReLU()
        self.activation1 = nn.Sigmoid()
    
    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x = F.dropout(x,0.3)
        x = self.activation(self.fc1(x))
        x = F.dropout(x,0.3)
        x = self.activation(self.fc2(x))
        x = F.dropout(x,0.3)
        x = self.activation(self.fc3(x))
        x = F.dropout(x,0.3)
        x = self.activation1(self.fc4(x))
        
        return x


def train(epoch,loader, gen, dis, opt_gen, opt_dis, criterion):
    # implementation of training method
    dis_loss_total = 0
    gen_loss_total = 0
    for i,(x,y) in enumerate(loader):
        # ===== train discriminator ==============
        y_real = Variable(torch.ones((x.shape[0], 1))).to(device)
        y_fake = Variable(torch.zeros((x.shape[0], 1))).to(device)
        
        disc_real = dis(x.to(device))
        # generating fake data
        z = Variable(torch.randn((x.shape[0], latent_dim)))
        fake = gen(z.to(device))
        disc_fake = dis(fake)
        
        opt_dis.zero_grad()
        # calculate loss
        disc_real_loss = criterion(disc_real, y_real)
        disc_fake_loss = criterion(disc_fake, y_fake)
        disc_loss = disc_real_loss + disc_fake_loss
        
        # update only discriminator module
        disc_loss.backward()
        opt_dis.step()
        
        # ======= train generator ============
        z = Variable(torch.randn((x.shape[0], latent_dim))).to(device)
        gen_fake = gen(z)
        disc_gen_fake = dis(gen_fake)
        
        opt_gen.zero_grad()
        gen_loss = criterion(disc_gen_fake, y_real)
        gen_loss.backward()
        opt_gen.step()
        
        
#         if i==0:
#             print('Gradients')
#             for name, param in dis.named_parameters():
#                 if 'bias' in name:
#                     print(name, param.grad[0], end=' ')
#                 else:
#                     print(name, param.grad[0,0], end=' ')
        
        dis_loss_total += disc_loss.cpu().data.numpy()*x.shape[0]
        gen_loss_total += gen_loss.cpu().data.numpy()*x.shape[0]
    
    dis_loss_total /= len(loader.dataset)
    gen_loss_total /= len(loader.dataset)
    return dis_loss_total, gen_loss_total


def plot(epoch, pred):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16,16))
    for i in range(25):
        ax = fig.add_subplot(5,5,i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
#         ax.title.set_text(str(y[i]))
    plt.savefig("./images/epoch_{}.jpg".format(epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def infer(epoch, z, gen, num_samples=25):
    pred = gen(z).cpu().data.numpy()
    pred = pred.reshape((num_samples, 1, 28, 28))
    plot(epoch, pred)


def save_model(gen, disc,epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    
    filename1 = './checkpoints/' + 'gen_{}.pt'.format(epoch)
    torch.save(gen.state_dict(), filename1)
    
    filename2 = './checkpoints/' + 'disc_{}.pt'.format(epoch)
    torch.save(disc.state_dict(), filename2)


if __name__ == "__main__":
	train_loader, test_loader = load_data() # load data & create dataloader object

	gen = Generator(input_dim=latent_dim).to(device) # crteating generator object
	dis = Discriminator().to(device) # crteating discriminator object
	
	if load_epoch > 0:
	    # if load_epoch > 0 then load saved model
	    gen.load_state_dict(torch.load('./checkpoints/gen_{}'.format(load_epoch)), map_location=torch.device('cpu'))
	    dis.load_state_dict(torch.load('./checkpoints/disc_{}'.format(load_epoch)), map_location=torch.device('cpu'))
	
	# initializig adam optimizer
	opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, weight_decay=0.001)
	opt_dis = optim.Adam(dis.parameters(), lr=learning_rate, weight_decay=0.001)
	
	# defining loss function
	criterion = nn.BCELoss()
	
	# taining loop
	gen_loss_list = []
	disc_loss_list = []
	if not os.path.isdir('./loss'):
	    os.mkdir('./loss')
	fixed_z = torch.randn((25, latent_dim)).to(device)
	
	for epoch in range(load_epoch+1, max_epoch):
	    gen.train()
	    dis.train()
	    disc_loss, gen_loss = train(epoch, train_loader, gen, dis, opt_gen, opt_dis, criterion)
	    gen_loss_list.append(gen_loss)
	    disc_loss_list.append(disc_loss)
	    print('Epoch: {}/{} Disc Loss: {} Gen Loss: {}'.format(epoch, max_epoch, disc_loss, gen_loss))
	    
	    save_model(gen,dis,epoch) # saving generator model
	    np.save('./loss/Generator_loss', np.array(gen_loss_list))
	    np.save('./loss/Discriminator_loss', np.array(disc_loss_list))
	    
	    with torch.no_grad():
	        gen.eval()
	        dis.eval()
	        infer(epoch, fixed_z, gen)