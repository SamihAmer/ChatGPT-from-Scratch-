import torch

'''
Complete this class by instantiating parameters called "self.weight" and "self.bias", and
use them to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		# TODO
		# initializes the weights Tensor with shape (output_size, input_size). Multiplying by 0.1 scales values down by 10x
		# so that our weights start with small values to make sure early outputs/gradients don't start too large
		initial_weights = 0.1 * torch.randn((output_size, input_size)) 
		
		# then we assign torch.nn.Parameter inside nn.Module to wrap the initial_weights Tensor and tell PyTorch:
		# "this is a learnable weight of our model"
		self.weight = torch.nn.Parameter(initial_weights)
		
		# now to implement the 'b' component of 'y=Wx+b' we know that b needs to add one offset per output dimension (neuron)
		# so this corresponds with a 1-D vector of length output_size and shape (output_size,)
		initial_bias = 0.1 * torch.randn(output_size)
		self.bias = torch.nn.Parameter(initial_bias) # here we use nn.Parameter to do the same that we did with the weight

	def forward(self, x):
		'''
		x is a tensor contain a batch of vectors, size (B, input_size).
		This should return a tensor of size (B, output_size).
		'''
		# TODO
		# Since x has a shape (B, input_size), we need W to be shape (input_size, output_size) for the matrix multiplication to work
		# This is why we transpose W to change the dimension and return a tensor with shape (B, output_size)

		# Lastly for the bias term, PyTorch "broadcasts" (i.e. expands) the 1-D bias tensor to be able do matrix addition 
		# with size (B, output_size)
		return x @ self.weight.T + self.bias
	
