import numpy as np
import torch
from tqdm import tqdm
from sampler import Sampler

# from module 6:
from gpt import GPTModel
from hftokenizer import HFTokenizer


'''
Example of using Sampler with our own GPT model trained on wikipedia data.

Just included for fun.
'''

# make sampling algo
samp = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

# load our model
model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000, max_seq_len=256)
model.load_state_dict(torch.load("./model_weights.pt"))
model.eval()

# make tokenizer
# also make sure to copy over the ./hftokenizer/ folder
# or edit hftokenizer.py to correctly point to it.
tokenizer = HFTokenizer()
tokenizer.load()

# =================================================================

# some text
intial_text = "Steve Jobs was"
token_ids = tokenizer.encode(intial_text)
token_ids = torch.tensor([token_ids]) # "batch" of 1
print(token_ids)

# generate N more tokens. We are not using kv cache or anything smart.
# This may be pretty slow.
for i in tqdm(range(100)):

	# pass tokens through the model to get logits
	output = model(token_ids)
	output = output[0,-1,:] # first seq in batch, last output

	# sample from the logits
	token_ids_np = token_ids.data.cpu().numpy()[0] # first seq
	tok = samp(output.data.cpu().numpy(), token_ids_np)

	# add the resulting token id to our list
	token_ids_np = np.append(token_ids_np, tok)
	token_ids = torch.from_numpy(token_ids_np)

	# add back a batch size of 1
	token_ids = token_ids[None,:]

	# if we generated a stop token, stop!
	# can comment this out to force the model to keep going.
	if tok == tokenizer.tokenizer.eos_token_id:
		break


token_ids = token_ids.data.cpu().numpy()[0]

# print out resulting ids
print(token_ids)

# print out the decoded text
print(tokenizer.decode(token_ids))