import torch

'''
Complete this class by instantiating a parameter called "self.weight", and
use it to complete the forward() method. You do not need to worry about backpropogation.
'''
class CustomEmbedding(torch.nn.Module):

	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		
		# Weight matrix: one row per token, one columb per embedding dimension
		# initialized with small random values
		self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))


	def forward(self, x):
		# x is a tensor of integers
		# Integer indexing into a 2-D tensor selects the corresponding rows
		# this s exactly the lookup table operation
		
		return self.weight[x]
	
"""
self.weight[x] works because PyTorch supports integer tensor indexing.
If self.weight has shape (V,D) and x is a 1-D tensor of length N, 
then self.weight[x] returns a tensor of shape (N,D) - one embedding vector
per token ID. this is functionally identical to torch.nn.Embedding 
"""
 
		