import torch
from linear import CustomLinear
from embedding import CustomEmbedding
from mha_mask import CustomMHA

'''
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use pytorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).

For layer norm, you just need to pass in D-model: self.norm1 = torch.nn.LayerNorm((d_model,))

'''
class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((d_model,)) # first layer norm over last dim
        self.mha = CustomMHA(d_model, n_heads)  # creates masked self-attention sublayer

        self.norm2 = torch.nn.LayerNorm((d_model)) # second layer norm before MLP
        self.ff1 = CustomLinear(d_model, 4 * d_model) # first MLP projection to expand feature capacity per token 
        self.act = torch.nn.ReLU()  # ReLU non-linear activation function
        self.ff2 = CustomLinear(4 * d_model, d_model)   # second MLP projection back to model width
        
        self.dropout = torch.nn.Dropout(0.1)  # regularization on MLP before residual add


    
    def forward(self, x):
        '''
        param x : (tensor) a tensor of size (batch_size, sequence_length, d_model)
        returns the computed output of the block with the same size.
        '''
        # takes x shaped (B,S,d_model) and applies attention sublayer and feed-forward sublayer
        x = x + self.mha(self.norm1(x))

        mlp_out = self.ff2(self.act(self.ff1(self.norm2(x))))

        x = x + self.dropout(mlp_out)

        return x   # returns updated tensor x ready for next decoder block 


'''
Create a full GPT model which has two embeddings (token and position),
and then has a series of transformer block instances (layers). Finally, the last 
layer should project outputs to size [vocab_size].
'''
class GPTModel(torch.nn.Module):

    
    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
        '''
        param d_model : (int) the size of embedding vectors and throughout the model
        param n_heads : (int) the number of attention heads, evenly divides d_model
        param layers : (int) the number of transformer decoder blocks
        param vocab_size : (int) the final output vector size
        param max_seq_len : (int) the longest sequence the model can process.
            This is used to create the position embedding- i.e. the highest possible
            position to embed is max_seq_len
        '''

        super().__init__()
        # hint: for a stack of N layers look at torch ModuleList or torch Sequential
        self.max_seq_len = max_seq_len
        self.token_embedding = CustomEmbedding(vocab_size, d_model) # map token ID to vector
        self.position_embedding = CustomEmbedding(max_seq_len, d_model) # map position ID to vector 

        self.blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(d_model, n_heads) for _ in range(layers)]   # create layer decoder blocks 
            # each block keeps shape (B, S, d_model)
        )

        self.output_proj = CustomLinear(d_model, vocab_size)  #final per token projection to logits
        #input (B,S,d_model) output (B,S,vocab_size)

    
    def forward(self, x):
        '''
        param x : (long tensor) an input of size (batch_size, sequence_length) which is
            filled with token ids

        returns a tensor of size (batch_size, sequence_length, vocab_size), the raw logits for the output
        '''
        # hint: x contains token ids, but you will also need to build a tensor of position ids here
        
        B,S = x.shape   # extracts batch size and sequence length

        if S > self.max_seq_len: # indexing to prevent learning beyond learned positional embedding table
            raise ValueError(f"sequence length {S} exceeds max sequence length {self.max_seq_len}")

       # position ID tensor 
        pos_ids = torch.arange(S, device = x.device, dtype = torch.long).unsqueeze(0).expand(B,S)

        h = self.token_embedding(x) + self.position_embedding(pos_ids) # combine token meaning and position info 

        for block in self.blocks:  # repeatedly transofmr h while keeping shape 
            h = block(h)

        # convert hidden state to vocabulary logits 
        logits = self.output_proj(h)  # (B,S, vocab_size)
        return logits    
            


if __name__ == "__main__":

    # example of building the model and doing a forward pass
    D = 128
    H = 8
    L = 4
    model = GPTModel(D, H, L, 1000, 512)
    B = 32
    S = 48 # this can be less than 512, it just cant be more than 512
    x = torch.randint(1000, (B, S))
    y = model(x) # this should give us logits over the vocab for all positions

    # should be size (B, S, 1000)
    print(y)
