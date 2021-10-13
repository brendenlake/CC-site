import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt    
from itertools import permutations
from data_loader import get_label_coding,load_shj_abstract,load_shj_images
from scipy.stats import sem

#
# PyTorch implementation of
#  	ALCOVE: An exemplar-based connectionist model of category learning (Kruschke, 1992)

#   Main script runs ALCOVE on stimuli of Shephard, Hovland, and Jenkins (1961)

#  There are a few modifications to Kruschke's original ALCOVE:
#    - with two classes, there is only one binary output
#    - rather the using Kruschke's loss, there are options to maximize
#    the log-likelihood directly (ll loss) or a version of the humble teacher (hinge loss)

class ALCOVE(nn.Module):

	def __init__(self, exemplars, c=6.5, phi=2.5):
		# Input
		#   exemplars: [ne x dim] rows are exemplars provided to model
		super(ALCOVE, self).__init__()
		self.ne = exemplars.size(0) # number of exemplars
		self.dim = exemplars.size(1) # stimulus dimension
		self.exemplars = exemplars # ne x dim

		# set attention weights to be uniform
		self.attn = torch.nn.Parameter(torch.ones((self.dim,1))/float(self.dim))

		# set association weights to zero
		self.w = torch.nn.Linear(self.ne,1,bias=False)
		self.w.weight = torch.nn.Parameter(torch.zeros((1,self.ne)))

		self.c = c # sharpness parameter (Kruschke uses 6.5 in SHJ simulations)
		self.phi = phi # temperature when making decisions; not included in loss (Kruschke uses 2.0)

	def forward(self,x):
		# Input
		#  x: [dim tensor] a single stimulus
		#
		# Output
		#  output : [tensor scalar] unnormalized log-score (before sigmoid)
		#  prob : [tensor scalar] sigmoid output
		x = x.view(-1,1) # dim x 1
		x_expand = x.expand((-1,self.ne)) # dim x ne
		x_expand = torch.t(x_expand) # ne x dim
		attn_expand = self.attn.expand((-1,self.ne)) # dim x ne
		attn_expand = torch.t(attn_expand) # ne x dim

		# memory/hidden layer is computes the similarity of stimulus x to each exemplar
		hidden = attn_expand * torch.abs(self.exemplars-x_expand) # ne x dim
		hidden = torch.sum(hidden,dim=1) # ne
		hidden = torch.exp(-self.c * hidden) # ne
		hidden = hidden.view((1,-1)) # 1 x ne
		
		# compute the output response
		output = self.w(hidden).view(-1) # tensor scalar
		prob = torch.sigmoid(self.phi*output) # tensor scalar
		return output,prob

class MLP(nn.Module):

	def __init__(self, nhid=8, phi=2.0):
		# Input
		#   exemplars: [ne x dim] rows are exemplars provided to model
		super(MLP, self).__init__()
		self.ne = exemplars.size(0) # number of exemplars
		self.dim = exemplars.size(1) # stimulus dimension
		self.nhid = nhid
		self.hid = torch.nn.Linear(self.dim,self.nhid)
		self.out = torch.nn.Linear(self.nhid,1)
		self.phi = phi

	def forward(self,x):
		# Input
		#  x: [dim tensor] a single stimulus
		#
		# Output
		#  output : [tensor scalar] unnormalized log-score (before sigmoid)
		#  prob : [tensor scalar] sigmoid output
		x = x.view(1,-1) # dim x 1
		x = self.hid(x)
		x = torch.tanh(x)
		output = self.out(x)
		prob = torch.sigmoid(self.phi*output) # tensor scalar
		return output,prob

def update_batch(net,exemplars,targets,loss,optimizer):
	# Update the weights using batch SGD for the entire set of exemplars
	#
	# Input
	#   exemplars: [ne x dim tensor] all stimuli/exempalrs in experiment 
	#   targets:   [ne tensor] classification targets (1/0 or 1/-1, depending on loss)
	#   loss: function handle
	#   optimizer : SGD optimizer
	net.zero_grad()
	net.train()
	n_exemplars = exemplars.size(0)
	out = torch.zeros(n_exemplars)
	for j in range(n_exemplars):
		out[j],_ = net.forward(exemplars[j])
	myloss = loss(out, targets)
	myloss.backward()
	optimizer.step()
	if model_type == 'alcove':
		net.attn.data = torch.clamp(net.attn.data, min=0.) # ensure attention is non-negative
	return myloss.cpu().item()

def evaluate(net,exemplars,targets):
	# Compute probability of getting each answer/exemplar right using sigmoid
	# 
	# Input
	#   exemplars: [ne x dim tensor] all stimuli/exempalrs in experiment 
	#   targets:   [ne tensor] classification targets (1/0 or 1/-1, depending on loss)
	#
	# Output
	#   mean probability of correct response
	#   mean accuracy when picking most likely response
	net.eval()
	n_exemplars = exemplars.size(0)
	v_acc = np.zeros(n_exemplars)
	v_prob = np.zeros(n_exemplars)
	for j in range(n_exemplars):
		out,prob = net.forward(exemplars[j])
		out = out.item() # logit
		prob = prob.item() # prob of decision 
		if targets[j].item()==POSITIVE:		
			v_prob[j] = prob
			v_acc[j] = out >= 0	
		elif targets[j].item()==NEGATIVE:
			v_prob[j] = 1-prob
			v_acc[j] = out < 0
	return np.mean(v_prob), 100.*np.mean(v_acc)

def HingeLoss(output, target):
	# Reinterpretation of Kruschke's humble teacher
	#  loss = max(0,1-output * target)
	#
	# Input
	#  output : 1D tensor (raw prediction signal)
	#  target : 1D tensor (must be -1. and 1. labels)
    hinge_loss = 1.-torch.mul(output, target)
    hinge_loss[hinge_loss < 0] = 0.
    return torch.sum(hinge_loss)

def train(exemplars,labels,num_epochs,loss_type,track_inc=5,verbose_params=False):
	# Train model on a SHJ problem
	# 
	# Input
	#   exemplars : [n_exemplars x dim tensor] rows are exemplars
	#   labels : [n_exemplars tensor] category labels	
	#   num_epochs : number of passes through exemplar set
	#   loss_type : either 'll' or 'hinge'
	#	track_inc : track loss/output at these intervals
	#   verbose_params : print parameters when you are done
	#
	# Output
	#    trackers for epoch index, probability of right response, accuracy, and loss
	#    each is a list with the same length
	n_exemplars = exemplars.size(0)
	
	if model_type == 'mlp':
		net = MLP()
	elif model_type == 'alcove':
		net = ALCOVE(exemplars)
	else:
		assert False

	if loss_type == 'll':
		loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
	elif loss_type == 'hinge':
		loss = HingeLoss
	else:
		assert False # undefined loss
	
	optimizer = optim.SGD(net.parameters(), lr=lr_association)
	if model_type == 'alcove':
		optimizer = optim.SGD([ {'params': net.w.parameters()}, {'params' : [net.attn], 'lr':lr_attn}], lr=lr_association)

	v_epoch = []
	v_loss = []
	v_acc = []
	v_prob = []
	for epoch in range(1,num_epochs+1):
		loss_epoch = update_batch(net,exemplars,labels,loss,optimizer)
		if epoch == 1 or epoch % track_inc == 0:
			test_prob,test_acc = evaluate(net,exemplars,labels)
			v_epoch.append(epoch)
			v_loss.append(loss_epoch / float(n_exemplars))
			v_acc.append(test_acc)
			v_prob.append(test_prob)
			print('  epoch ' + str(epoch) + "; train loss " + str(round(v_loss[-1],4)))

	if model_type == 'alcove' and verbose_params:
		print("Attention weights:")
		print(np.transpose(np.round(net.attn.data.numpy(),3)))
		print("Category associations:")
		print(np.round(net.w.weight.data.numpy(),3))


	return v_epoch,v_prob,v_acc,v_loss

if __name__ == "__main__":

	num_epochs = 50 # number of passes through exemplars
	model_type = 'alcove' # 'alcove' or 'mlp'
	data_type = 'abstract' # 'abstract' (binary representation) or 'images' (pixels)
	loss_type = 'hinge' # 'll' (log-likelihood) or 'hinge' (version of humble teacher)
	lr_association = 0.03 # learning rate for association weights
	lr_attn = 0.0033 # learning rate for attention weights
	ntype = 6 # number of types in SHJ
	viz_se = False # visualize standard error in plot

	POSITIVE,NEGATIVE = get_label_coding(loss_type)
	if data_type == 'abstract':
		list_perms = [(0,1,2)]
		list_exemplars = []
		for p in list_perms:
			exemplars,labels_by_type = load_shj_abstract(loss_type,p) 
				# [n_exemplars x dim tensor],list of [n_exemplars tensor]		
			list_exemplars.append(exemplars)
	elif data_type == 'images':
		list_perms = list(permutations([0,1,2])) # ways of assigning abstract dimensions to visual ones
		list_exemplars = []
		for p in list_perms:
			exemplars,labels_by_type = load_shj_images(loss_type,p)
				# [n_exemplars x dim tensor],list of [n_exemplars tensor]
			list_exemplars.append(exemplars)
	else:
		assert False
	dim = list_exemplars[0].size(1)
	print("Data loaded with " + str(dim) + " dimensions.")

	# Run ALCOVE on each SHJ problem
	list_trackers = []
	for pidx,exemplars in enumerate(list_exemplars): # all permutations of stimulus dimensions
		tracker = []
		print('Permutation ' + str(pidx))
		for mytype in range(1,ntype+1): # from type I to type VI
			print('  Training on type ' + str(mytype))
			labels = labels_by_type[mytype-1]
			v_epoch,v_prob,v_acc,v_loss = train(exemplars,labels,num_epochs,loss_type)
			tracker.append((v_epoch,v_prob,v_acc,v_loss))
			print("")
		list_trackers.append(tracker)

	A = np.array(list_trackers) # nperms x ntype x 4 tracker types x n_iters
	M = np.mean(A,axis=0) # ntype x 4 tracker types x n_iters
	SE = sem(A,axis=0) # ntype x 4 tracker types x n_iters

	plt.figure(1)
	for i in range(ntype): 
		if viz_se:
			plt.errorbar(M[i,0,:],M[i,1,:],yerr=SE[i,1,:],linewidth=4./(i+1))
		else:
			plt.plot(M[i,0,:],M[i,1,:],linewidth=4./(i+1))
	plt.xlabel('Block')
	plt.ylabel('Probability correct')
	plt.legend(["Type " + str(s) for s in range(1,7)])
	
	plt.figure(2)
	for i in range(ntype):
		if viz_se:
			plt.errorbar(M[i,0,:],M[i,3,:],yerr=SE[i,3,:],linewidth=4./(i+1))  # v is [tracker type x n_iters]
		else:
			plt.plot(M[i,0,:],M[i,3,:],linewidth=4./(i+1))  # v is [tracker type x n_iters]
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(["Type " + str(s) for s in range(1,7)])
	plt.show()
