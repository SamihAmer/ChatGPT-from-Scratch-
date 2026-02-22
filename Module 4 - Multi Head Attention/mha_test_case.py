import torch
import math
# this is a test case to help you debug your implementation.
# your class up here
class CustomMHA(torch.nn.Module):

	'''
	param d_model : (int) the length of vectors used in this model
	param n_heads : (int) the number of attention heads. You can assume that
		this even divides d_model.
	'''
	def __init__(self, d_model, n_heads):
		super().__init__()
		# TODO
		# please name your parameters "self.W_qkv" and "self.W_o" to aid in grading
		# self.W_qkv should have shape (3D, D)
		# self.W_o should have shape (D,D)
		self.d_model = d_model
		self.n_heads = n_heads
		W_qkv = torch.randn(3*d_model,d_model)
		W_o = torch.randn(d_model, d_model)
		self.W_qkv = torch.nn.Parameter(W_qkv)
		self.W_o = torch.nn.Parameter(W_o)
	'''
	param x : (tensor) an input batch, with size (batch_size, sequence_length, d_model)
	returns : a tensor of the same size, which has had MHA computed for each batch entry.
	'''
	def forward(self, x):
		# step 1: matrix multiply x * W_qkv.T to get tensor T size (Sx3D) ~ (B,S,3D)
		T = x @ self.W_qkv.T
		print(T.shape)

		# step 2: slice T to get queries, keys, and values of shape (B,S,D)
		Q, K, V = torch.chunk(T, 3, dim = -1)
		print(Q.shape, K.shape, V.shape)

		# step 3: reshape (B,S,D) to (B,S,h,D/h) 
		B = x.shape[0]
		S = x.shape[1]
		h = self.n_heads
		D = self.d_model
		
		# torch.Tensor.view() returns a new tensor but with different shape  
		Q = Q.view(B, S, h, D//h)
		K = K.view(B, S, h, D//h)
		V = V.view(B, S, h, D//h)
		
		print(Q.shape, K.shape, V.shape)

		# step 3: transpose to get (B,h,S,D/h)
		Q = Q.transpose(1,2)  # torch.transpose lets us swap dimensions in a tensor
		K = K.transpose(1,2)
		V = V.transpose(1,2)

		print(Q.shape, K.shape, V.shape)

		# step 4: weight computation 
		weight = (Q @ K.transpose(2,3)) / math.sqrt(D/h)
		print(weight.shape)

		# step 5: apply softmax 
		weight = torch.nn.functional.softmax(weight, dim = -1) 

		# step 6: matrix multiply by Values
		attn = weight @ V
		print(attn.shape) # should be (B,h,S,D/h)

		# step 7: reshape attn back to (B,S,D)
		attn = attn.transpose(1,2) # make back to (B,S,h,D/h)
		# Here, I am citing the advice from Peter Bloem's Blog Post 
		# Torch.View will require contiguous memory, and transpose makes the tensor non-contiguous
		# Torch.reshape() copies the data directly and doesn't require the data to be contiguous
		# so using reshape lets us avoid calling .contiguous 
		attn = torch.reshape(attn, (B,S,D))  

		# step 8: matrix multiply against output projection W_o
		return attn @ self.W_o.T
	

if __name__ == "__main__":


	import torch
	import numpy as np

	D = 6
	H = 2
	mha = CustomMHA(D,H)

	# make some fixed weights
	# this just makes a really long 1-D np array and then reshapes it into the size we need
	tensor1 = torch.tensor(np.reshape(np.linspace(-2.0, 1.5, D*D*3), (D*3,D))).to(torch.float32)
	tensor2 = torch.tensor(np.reshape(np.linspace(-1.0, 2.0, D*D), (D,D))).to(torch.float32)
	
	# copy these into our MHA weights, so we don't need to worry about random initializations for testing
	mha.W_qkv.data = tensor1
	mha.W_o.data = tensor2

	# make an input tensor
	B = 2
	S = 3
	x = torch.tensor(np.reshape(np.linspace(-1.0, 0.5, B*S*D), (B,S,D))).to(torch.float32)

	# run
	y1 = mha(x)
	print(y1.shape)
	print(y1)

	'''
	Should print out:

	torch.Size([2, 3, 6])
	tensor([[[ 17.2176,   5.5439,  -6.1297, -17.8034, -29.4771, -41.1508],
         [ 17.4543,   5.5927,  -6.2688, -18.1304, -29.9920, -41.8536],
         [ 17.6900,   5.6398,  -6.4105, -18.4607, -30.5110, -42.5612]],

        [[ -1.3639,  -0.1192,   1.1256,   2.3703,   3.6151,   4.8598],
         [ -5.5731,  -1.9685,   1.6361,   5.2407,   8.8453,  12.4499],
         [ -5.6875,  -2.0716,   1.5444,   5.1603,   8.7762,  12.3922]]],
       grad_fn=<UnsafeViewBackward0>)

	'''