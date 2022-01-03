import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import gradient_penalty

def plot(epoch, pred):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16,16))
    for i in range(25):
        ax = fig.add_subplot(5,5,i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
    plt.savefig("./images/epoch_{}.jpg".format(epoch))
    plt.close()


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURE_DISC = 64
FEATURE_GEN = 64
CRITIC_ITRERATIONIS = 5
LAMBDA_GP = 10 


transforms = transforms.Compose(
	[
		transforms.Resize(IMAGE_SIZE),
		transforms.ToTensor(),
		transforms.Normalize(
			[0.5 for _ in range(CHANELS_IMG)], [0.5 for _ in range(CHANELS_IMG)]
		),
	]
)


dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANELS_IMG, FEATURE_GEN).to(device)
critic = Discriminator(CHANELS_IMG, FEATURE_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)


opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fised_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
	for batch_idx, (real, _) in enumerate(dataloader):
		real = real.to(device)
		BATCH_SIZE = real.shape[0]
		for _ in range(CRITIC_ITRERATIONIS):
			noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
			fake = gen(noise)
			critic_real = critic(real).reshape(-1)
			critic_fake = critic(fake).reshape(-1)
			gp = gradient_penalty(critic, real, fake, device=device)
			loss_critic = (
					-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
				)
			critic.zero_grad()
			loss_critic.backward(retain_graph=True)
			opt_critic.step()

			# for p in critic.parameters():
			# 	p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

		output = critic(fake).reshape(-1)
		loss_gen = -torch.mean(output)
		gen.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		if batch_idx%100 == 0:
			print(f"Epoch: [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)}\
					Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}")
			with torch.no_grad():
				fake = gen(fised_noise)
				plot(epoch*1000+batch_idx, fake[:25].cpu().data.numpy())
