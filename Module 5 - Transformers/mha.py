import torch
import math

'''
Complete this module such that it computes queries, keys, and values,
computes attention, and passes through a final linear operation W_o.

You do NOT need to apply a causal mask (we will do that next week).
If you don't know what that is, don't worry, we will cover it next lecture.

Be careful with your tensor shapes! Print them out and try feeding data through
your model. Make sure it behaves as you would expect.
'''
class CustomMHA(torch.nn.Module):

	'''
	param d_model : (int) the length of vectors used in this model
	param n_heads : (int) the number of attention heads. You can assume that
		this even divides d_model.
	'''
	def __init__(self, d_model, n_heads):
		super().__init__()
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
		#print(T.shape)

		# step 2: slice T to get queries, keys, and values of shape (B,S,D)
		Q, K, V = torch.chunk(T, 3, dim = -1)
		#print(Q.shape, K.shape, V.shape)

		# step 3: reshape (B,S,D) to (B,S,h,D/h) 
		B = x.shape[0]
		S = x.shape[1]
		h = self.n_heads
		D = self.d_model
		
		# torch.Tensor.view() returns a new tensor but with different shape  
		Q = Q.view(B, S, h, D//h)
		K = K.view(B, S, h, D//h)
		V = V.view(B, S, h, D//h)
		
		#print(Q.shape, K.shape, V.shape)

		# step 3: transpose to get (B,h,S,D/h)
		Q = Q.transpose(1,2)  # torch.transpose lets us swap dimensions in a tensor
		K = K.transpose(1,2)
		V = V.transpose(1,2)

		#print(Q.shape, K.shape, V.shape)

		# step 4: weight computation 
		weight = (Q @ K.transpose(2,3)) / math.sqrt(D/h)
		#print(weight.shape)

		# step 5: apply softmax 
		weight = torch.nn.functional.softmax(weight, dim = -1) 

		# step 6: matrix multiply by Values
		attn = weight @ V
		#print(attn.shape) # should be (B,h,S,D/h)

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

	# example of building and running this class
	mha = CustomMHA(128,8)

	# 32 samples of length 6 each, with d_model at 128
	x = torch.randn((32,6,128))
	y = mha(x)
	print(x.shape, y.shape) # should be the same