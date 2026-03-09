
import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

	def __init__(
		self,
		top_k=None,
		top_p=None,
		frequency_penalty=1.0,
		presence_penalty=1.0
	):
		'''
		param top_k : (None or int)
			If specified, only the top k logits should be used during sampling
			If this is specified, top_p should be None

		param top_p : (None or int)
			If specified, only the logits representing the probability mass p should be used during sampling.
			Or, if the top token has mass greater than p, the top token is returned.
			If this is specified, top_k should be None

		If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

		param frequency_penalty : (float)
			A penalty applied to tokens that have previously occured in the sequence. Along with
			presence_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.

		param presence_penalty : (float)
			A penalty applied to tokens IF they have previously occured in the sequence. Along with
			frequency_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.
		'''
		# Store parameters
		self.top_k = top_k
		self.top_p = top_p
		self.frequency_penalty = frequency_penalty
		self.presence_penalty = presence_penalty

		# Validate that top_k and top_p are not both specified
		if top_k is not None and top_p is not None:
			raise ValueError("top_k and top_p are mutually exclusive; only one can be specified.")


	def make_token_distribution(self, raw_unsorted_logits, previous_token_ids):
		'''
		param: raw_unsorted_logits (float numpy array)
			A one dimensional list of logits representing an unnormalized distribution over next tokens
			These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

		param: previous_token_ids (int numpy array)
			A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

		returns:
			- the final probability distribution that this token is sampled from
			It should be returned back to token-id order (unsorted order) before returning.
		'''

		logits = np.array(raw_unsorted_logits, dtype=np.float64)
		vocab_size = len(logits)

		# ---- Step 1: Build per-token temperature values ----
		# Start with temperature k = 1.0 for every token in the vocabulary
		temperatures = np.ones(vocab_size, dtype=np.float64)

		# ---- Step 2: Apply frequency penalty ----
		# Frequency penalty increases the temperature proportionally to how many
		# times a token has appeared in the previous sequence.
		# k += occurrences * (frequency_penalty - 1.0)
		if self.frequency_penalty != 1.0:
			# Count occurrences of each token id in previous_token_ids
			for tok_id in previous_token_ids:
				if 0 <= tok_id < vocab_size:
					temperatures[tok_id] += (self.frequency_penalty - 1.0)

		# ---- Step 3: Apply presence penalty ----
		# Presence penalty increases the temperature if the token has appeared
		# at all in the previous sequence (binary: occurred or not).
		# k += (presence_penalty - 1.0) if token has occurred
		if self.presence_penalty != 1.0:
			# Find unique token ids that have occurred
			unique_previous = set(previous_token_ids)
			for tok_id in unique_previous:
				if 0 <= tok_id < vocab_size:
					temperatures[tok_id] += (self.presence_penalty - 1.0)

		# ---- Step 4: Shift logits to be non-negative ----
		# The penalty (dividing by temperature > 1) only works correctly when
		# logits are positive. Subtracting the minimum ensures all are >= 0.
		logits = logits - np.min(logits)

		# ---- Step 5: Apply per-token temperatures and softmax ----
		# Divide each logit by its temperature, then apply softmax
		logits = logits / temperatures
		# Softmax: subtract max for numerical stability, then exponentiate
		logits = logits - np.max(logits)
		exp_logits = np.exp(logits)
		probs = exp_logits / np.sum(exp_logits)

		# ---- Step 6: Sort by probability (descending) ----
		# We sort in ascending order with argsort, then reverse to get descending
		indices = np.argsort(probs)[::-1]  # indices that sort descending
		undo_indices = np.argsort(indices)  # to revert back to original order
		sorted_probs = probs[indices]

		# ---- Step 7: Apply top-k or top-p cutoff ----
		if self.top_k is not None:
			# Top-K: keep only the top k tokens, zero out the rest
			cutoff = min(self.top_k, vocab_size)
			sorted_probs[cutoff:] = 0.0

		elif self.top_p is not None:
			# Top-P (nucleus sampling): keep the smallest set of tokens whose
			# cumulative probability mass is >= top_p.
			# Always keep at least the top token.
			cumsum = np.cumsum(sorted_probs)
			# Find where cumulative sum first exceeds top_p
			cutoff_idx = np.searchsorted(cumsum, self.top_p, side='left') + 1
			cutoff_idx = max(1, min(cutoff_idx, vocab_size))
			sorted_probs[cutoff_idx:] = 0.0

		# If both are None, we sample from the full distribution (no cutoff needed)

		# ---- Step 8: Renormalize ----
		total = np.sum(sorted_probs)
		if total > 0:
			sorted_probs = sorted_probs / total
		else:
			# Fallback: uniform over all tokens (shouldn't happen normally)
			sorted_probs = np.ones(vocab_size) / vocab_size

		# ---- Step 9: Revert to original token-id ordering ----
		final_probs = sorted_probs[undo_indices]

		return final_probs



	#==========================
	# for actually sampling the distribution
	def sample_one_token(self, raw_unsorted_logits, previous_token_ids):
		probs = self.make_token_distribution(raw_unsorted_logits, previous_token_ids)
		return np.random.choice(np.arange(len(raw_unsorted_logits)), p=probs)

	# for convenience, this is also callable
	def __call__(self, raw_unsorted_logits, previous_token_ids):
		return self.sample_one_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data, keeping everything in token ids

    sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

    sequence = [1,2,3,4,5]

    for i in range(10):
    	# fake logits for a vocab of size 500
    	logits = np.random.randn(500)

    	# get next token in sequence
    	next_token = sampler(logits, sequence)
    	sequence.append(next_token)

    print(sequence)