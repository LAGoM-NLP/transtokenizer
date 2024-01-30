import os
import json

import torch
import torch.nn as nn

import transformers

OLD_LANGUAGE='en'
NEW_LANGUAGE='tt'
MAPPING_FILE='alignments/NLLB.en-tt.mistralai--Mistral-7B-v0.1-tokenizers--tt-WORDS.token_mapping.json'

OLD_TOKENIZER='mistralai/Mistral-7B-v0.1'
NEW_TOKENIZER='tokenizers/tt'

OLD_MODEL_NAME=OLD_TOKENIZER
NEW_MODEL_NAME=OLD_MODEL_NAME.replace('/','--')+'--'+NEW_LANGUAGE

# load tokenizers for the two models
old_tokenizer = transformers.AutoTokenizer.from_pretrained(OLD_TOKENIZER)
new_tokenizer = transformers.AutoTokenizer.from_pretrained(NEW_TOKENIZER)

# load the old model
model = transformers.AutoModelForCausalLM.from_pretrained(OLD_MODEL_NAME)

# get the mapping between the two tokenizers
from typing import Tuple
mapping : "list[Tuple[str, list[Tuple[str, float]]]]" = []
with open(MAPPING_FILE, 'r') as f:
    mapping : "list[Tuple[str, list[Tuple[str, float]]]]" = json.load(f)

# disable gradient computation temporarily
with torch.no_grad():

    # get the embeddings of the OLM model
    old_embeddings = model.get_input_embeddings()
    old_output_embeddings = model.get_output_embeddings()

    # change the tokenizer of the OLM model to the one of the RobBERT model, and reinitialize the embeddings
    model.resize_token_embeddings(1) # this is a hack to make the model forget its old tokenizer
    model.resize_token_embeddings(len(new_tokenizer)) # this is the actual call to change the tokenizer
    new_embeddings = model.get_input_embeddings()
    new_output_embeddings = model.get_output_embeddings()
    model.config.vocab_size = len(new_tokenizer)
    model.config.pad_token_id = new_tokenizer.pad_token_id
    model.config.bos_token_id = new_tokenizer.bos_token_id
    model.config.eos_token_id = new_tokenizer.eos_token_id
    model.config.unk_token_id = new_tokenizer.unk_token_id
    model.config.sep_token_id = new_tokenizer.sep_token_id
    model.config.cls_token_id = new_tokenizer.cls_token_id
    model.config.mask_token_id = new_tokenizer.mask_token_id
    model.config.additional_special_tokens_ids = new_tokenizer.additional_special_tokens_ids
    model.config.tokenizer_class = new_tokenizer.__class__.__name__

    # for each token in the new tokenizer, find the corresponding tokens in the old tokenizer, and average their embeddings
    from tqdm import tqdm
    from functools import reduce

    for new_id in tqdm(range(len(new_tokenizer))):

        #new_token = new_tokenizer.convert_ids_to_tokens(new_id)
        old_tokens = mapping[new_id][1] # list of (ids,weight) in the old tokenizer

        # sort the list such that the smallest weights come first (for numerical stability)
        old_tokens = sorted(old_tokens, key=lambda x: x[1])

        # map tokens to their ids
        old_ids = [(old_tokenizer.convert_tokens_to_ids(old_token), weight) for old_token, weight in old_tokens]

        # we use a weighted average, where the first token in the list has 0.4 weight, the second 0.2, and the remaining 0.4 are equally distributed among all tokens (including the first two)
        if len(old_ids) > 1:
            new_embeddings.weight.data[new_id] = reduce(lambda a, b: a.add_(b), [old_embeddings.weight.data[old_id]*weight for old_id, weight in old_ids])
            new_output_embeddings.weight.data[new_id] = reduce(lambda a, b: a.add_(b), [old_output_embeddings.weight.data[old_id]*weight for old_id, weight in old_ids])
        elif len(old_ids) == 1:
            new_embeddings.weight.data[new_id] = old_embeddings.weight.data[old_ids[0][0]]
            new_output_embeddings.weight.data[new_id] = old_output_embeddings.weight.data[old_ids[0][0]]
        else: # use the unknown token embedding if the token is not in the old tokenizer
            new_embeddings.weight.data[new_id] = old_embeddings.weight.data[old_tokenizer.unk_token_id]
            new_output_embeddings.weight.data[new_id] = old_output_embeddings.weight.data[old_tokenizer.unk_token_id]
        

# save the model
os.makedirs('output', exist_ok=True)
model.save_pretrained('output/'+NEW_MODEL_NAME)
new_tokenizer.save_pretrained('output/'+NEW_MODEL_NAME)
