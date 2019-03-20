import numpy as np
import matplotlib.pyplot as plt

def naivegan():
	D,G = [], []
	fid_D = open('loss_curve_data/gan_D_loss.txt','r')
	fid_G = open('loss_curve_data/gan_G_loss.txt','r')
	lines = fid_D.readlines()
	D = [float(i.strip()) for i in lines]
	D = np.array(D)
	lines = fid_G.readlines()
	G = [float(i.strip()) for i in lines]
	G = np.array(G)

	#plt.figure()
	plt.plot(D,label = 'Loss of D')
	plt.plot(G, label = 'Loss of G')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss of Basic GAN')
	plt.savefig('loss_curve_fig/naivegan.png')
	plt.show()
	fid_D.close()
	fid_G.close()
	return

def wgan():
	fid_D = open('loss_curve_data/wgan_D_loss.txt','r')
	fid_G = open('loss_curve_data/wgan_G_loss.txt','r')
	fid_fake = open('loss_curve_data/wgan_D_loss_fake.txt','r')
	fid_real = open('loss_curve_data/wgan_D_loss_real.txt','r')
	D = wgan_helper(fid_D)
	G = wgan_helper(fid_G)
	Df =wgan_helper(fid_fake)
	Dr =wgan_helper(fid_real)
	xaxis = [i for i in range(len(D))]
	xaxis = np.array(xaxis)

	plt.scatter(xaxis,D,label = 'Loss of D',s=0.5)
	plt.scatter(xaxis,G, label = 'Loss of G',s=0.5)
	plt.ylim(-1, 0.5)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss of WGAN with Deep Convolutional Network Embedded')
	plt.savefig('loss_curve_fig/wgan_DG.png')
	plt.show()
	plt.clf()

	plt.scatter(xaxis,Df,label = 'Fake Loss of D',s=0.5)
	plt.scatter(xaxis,Dr, label = 'Real Loss of D',s=0.5)
	plt.ylim(-0.5, 0.5)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Fake Loss vs Real Loss of WGAN with Deep Convolutional Network Embedded')
	plt.savefig('loss_curve_fig/wgan_FR.png')
	plt.show()

	fid_D.close()
	fid_G.close()
	fid_fake.close()
	fid_real.close()
	
	return

def wgan_helper(fid):
	lines = fid.readlines()
	data = []
	for s in lines:
		cur_num = ''
		i = 0
		while i < len(s):
			cur_num = ''
			while s[i].isdigit() or s[i] == '.' or s[i] == '-':
				cur_num += s[i]
				i += 1
			if len(cur_num) != 0:
				data.append(float(cur_num))
				break
			i += 1
	return np.array(data)




def dcgan():
	D,G = [], []
	fid = open('loss_curve_data/dcgan_loss.txt','r')
	lines = fid.readlines()
	for s in lines:
		i = 0
		while i < len(s):
			cur_num = ''
			while i > 17 and (s[i].isdigit() or s[i] == '.'):
				cur_num += s[i]
				i += 1
			if len(cur_num) != 0:
				if len(D) == len(G):
					#print(cur_num)
					D.append(float(cur_num))
				else:
					G.append(float(cur_num))
			i += 1
	D = np.array(D)
	G = np.array(G)

	#plt.figure()
	plt.plot(D,label = 'Loss of D')
	plt.plot(G, label = 'Loss of G')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss of DCGAN')
	plt.savefig('loss_curve_fig/dcgan.png')
	plt.show()
	fid.close()
	return

def naivewgan():
	D,G = [], []
	fid_D = open('loss_curve_data/naive_wgan_D_loss.txt','r')
	fid_G = open('loss_curve_data/naive_wgan_G_loss.txt','r')
	lines = fid_D.readlines()
	D = [float(i.strip()) for i in lines]
	D = np.array(D)
	lines = fid_G.readlines()
	G = [float(i.strip()) for i in lines]
	G = np.array(G)

	#plt.figure()
	plt.plot(D,label = 'Loss of D')
	plt.plot(G, label = 'Loss of G')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Loss of WGAN with Simple Embedded Network')
	plt.savefig('loss_curve_fig/naivewgan.png')
	plt.show()
	fid_D.close()
	fid_G.close()
	return
#naivegan()
#dcgan()
#wgan()
naivewgan()

