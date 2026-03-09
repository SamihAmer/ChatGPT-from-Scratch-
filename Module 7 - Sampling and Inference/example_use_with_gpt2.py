import numpy as np
import torch
from tqdm import tqdm

'''
This script uses the Sampler class to sample text from GPT2-small.

You do not need to utilize this script for the assignment, but it may be
helpful or informative to see your Sampler applied to a real model.

Note this will download about 550MB of parameter data so you can run gpt2.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from sampler import Sampler

samp = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

# download gpt2 and the associated tokenizer
# you can try other sizes of gpt2 if you want, i.e. "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.eval()

# some text
intial_text = "Thomas Jefferson was the"
token_ids = tokenizer.encode(intial_text, return_tensors='pt')
print(token_ids)

# generate N more tokens. We are not using kv cache or anything smart.
# This may be pretty slow.
for i in tqdm(range(50)):

	# pass tokens through the model to get logits
	output = model(token_ids)["logits"][0,-1,:] # first seq in batch, last output

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
	if tok == tokenizer.eos_token_id:
		break

# grab the final sequence as numpy array
token_ids = token_ids.data.cpu().numpy()[0]

# print out resulting ids
print(token_ids)

# print out the decoded text
print(tokenizer.decode(token_ids))