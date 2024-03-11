
from collections import defaultdict
import os
from pathlib import Path
from typing import Iterator, Tuple
import regex as re
from tqdm import tqdm
from langcodes import Language

import torch
import torch.nn as nn
import math
import numpy as np
from collections import defaultdict

import transformers
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset


ALIGNMENT_UNIT = "WORDS"  # "TOKENS" or "WORDS"
MIN_COUNT_REQUIRED_FOR_CONSIDERATION = 20

home_path = os.environ['TT_HOME'] if "TT_HOME" in os.environ else Path("../notebooks/")


def get_dataset_iterator(dataset_name: str, source_language: str, target_language: str):
    """Utility function to get the iterable of a dataset, mapping different dataset conventions.

    Currently supported:
    - open_subtitles
    - allenai/nllb
    """

    if dataset_name == "open_subtitles":
        dataset = load_dataset(dataset_name, lang1=source_language, lang2=target_language, streaming=True)

        # wrap the dataset iterator so it returns a tuple of the source and target sentences
        class DatasetWrapper(Iterator):
            def __init__(self, dataset):
                self.dataset = iter(dataset['train'])

            def __next__(self):
                example = next(self.dataset)
                return example['translation'][source_language], example['translation'][target_language]

    elif dataset_name == "allenai/nllb":
        # map 2-letter language codes to 3-letter language codes (ISO 639-2 Cod from ISO 639-1 Code)
        dataset = load_dataset(
            dataset_name,
            f"{Language.get(source_language).to_alpha3()}_Latn-{Language.get(target_language).to_alpha3()}_Latn",
            split='train',
            streaming=True,
        )

        # wrap the dataset iterator so it returns a tuple of the source and target sentences
        class DatasetWrapper(Iterator):
            def __init__(self, dataset):
                self.dataset = iter(dataset)

            def __next__(self):
                example = next(self.dataset)
                return (
                    example['translation'][f"{Language.get(source_language).to_alpha3()}_Latn"],
                    example['translation'][f"{Language.get(target_language).to_alpha3()}_Latn"],
                )

    return DatasetWrapper(dataset)


def create_aligned_corpus(
    source_language: str,
    target_language: str,
    source_tokenizer: str,
    target_tokenizer: str,
    corpus_list: list = ['open_subtitles', 'allenai/nllb'],
):
    corpus_list_description = "_".join(corpus_list).replace("/", "--")
    OLD_TOKENIZER_FRIENDLY_NAME = source_tokenizer.replace('/', '--')
    NEW_TOKENIZER_FRIENDLY_NAME = target_tokenizer.replace('/', '--')

    # load tokenizers for the two models
    old_tokenizer = transformers.AutoTokenizer.from_pretrained(source_tokenizer)
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(target_tokenizer)

    # save the vocabularies in a set for improved performance
    old_tokenizer_vocab = set(old_tokenizer.vocab.keys())
    new_tokenizer_vocab = set(new_tokenizer.vocab.keys())

    # determine the tokenizer settings
    OLD_TOKENIZER_1ST_PREFIX = old_tokenizer.convert_ids_to_tokens(
        old_tokenizer.encode(" a", add_special_tokens=False)[0]
    ).rstrip("a")
    NEW_TOKENIZER_1ST_PREFIX = new_tokenizer.convert_ids_to_tokens(
        new_tokenizer.encode(" a", add_special_tokens=False)[0]
    ).rstrip("a")
    OLD_TOKENIZER_2ND_PREFIX = old_tokenizer.convert_ids_to_tokens(
        old_tokenizer.encode("aaaaaaaaaaaaaaaaaaaaaa", add_special_tokens=False)[1]
    ).rstrip('a')
    NEW_TOKENIZER_2ND_PREFIX = new_tokenizer.convert_ids_to_tokens(
        new_tokenizer.encode("aaaaaaaaaaaaaaaaaaaaaa", add_special_tokens=False)[1]
    ).rstrip('a')

    if os.path.exists(
        f'{home_path}/alignments/{corpus_list_description}.{source_language}-{target_language}.{OLD_TOKENIZER_FRIENDLY_NAME}-{NEW_TOKENIZER_FRIENDLY_NAME}-{ALIGNMENT_UNIT}.fast_align.tsv'
    ):
        print(f'data already aligned')
    else:
        out_path = f'{home_path}/alignments/{corpus_list_description}.{source_language}-{target_language}.{OLD_TOKENIZER_FRIENDLY_NAME}-{NEW_TOKENIZER_FRIENDLY_NAME}-{ALIGNMENT_UNIT}.moses'

        if os.path.exists(out_path):
            print(f'data already preprocessed for fast_align')
        else:
            os.makedirs(f'{home_path}/alignments', exist_ok=True)
            for corpus in corpus_list:
                with open(out_path, 'a') as f:
                    dataset = get_dataset_iterator(corpus, source_language, target_language)
                    for line_source, line_target in tqdm(dataset):
                        if ALIGNMENT_UNIT == 'WORDS':
                            # merging tokens from word units, for a better alignment
                            line1 = re.sub(
                                r'(?!'
                                + OLD_TOKENIZER_1ST_PREFIX
                                + r')(\p{L})[ ](?!'
                                + OLD_TOKENIZER_1ST_PREFIX
                                + r')(?='
                                + OLD_TOKENIZER_2ND_PREFIX
                                + r'\p{L})',
                                r'\1—',
                                ' '.join(old_tokenizer.tokenize(line_source.strip())),
                            )
                            line2 = re.sub(
                                r'(?!'
                                + NEW_TOKENIZER_1ST_PREFIX
                                + r')(\p{L})[ ](?!'
                                + NEW_TOKENIZER_1ST_PREFIX
                                + r')(?='
                                + NEW_TOKENIZER_2ND_PREFIX
                                + r'\p{L})',
                                r'\1—',
                                ' '.join(new_tokenizer.tokenize(line_target.strip())),
                            )

                        f.write(line1.strip() + ' ||| ' + line2.strip() + '\n')

    return f'{home_path}/alignments/{corpus_list_description}.{source_language}-{target_language}.{OLD_TOKENIZER_FRIENDLY_NAME}-{NEW_TOKENIZER_FRIENDLY_NAME}-{ALIGNMENT_UNIT}.moses'


def align(corpus: str, fast_align_path: str = "fast_align") -> str:
    if ".moses" not in corpus:
        raise ValueError("The input file should be a moses file")

    # check if fast_align is installed
    if os.system(f"{fast_align_path} -h") != 256:
        raise ValueError(
            "fast_align is not installed. Please install it from https://github.com/FremyCompany/fast_align"
        )

    # call fast_align
    os.system(f'{fast_align_path} -i {corpus} -I 7 -p {corpus.replace(".moses", ".fast_align.tsv")} > /dev/null')

    return corpus.replace(".moses", ".fast_align.tsv")


def map_tokens(
    mapped_tokens_file: str,
    source_tokenizer: str,
    target_tokenizer: str,
):
    print("Mapping tokens")
    OLD_TOKENIZER_FRIENDLY_NAME = source_tokenizer.replace('/', '--')
    NEW_TOKENIZER_FRIENDLY_NAME = target_tokenizer.replace('/', '--')

    # load tokenizers for the two models
    old_tokenizer = transformers.AutoTokenizer.from_pretrained(source_tokenizer)
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(target_tokenizer)

    # save the vocabularies in a set for improved performance
    old_tokenizer_vocab = set(old_tokenizer.vocab.keys())
    new_tokenizer_vocab = set(new_tokenizer.vocab.keys())

   
    tokenized_possible_translations = defaultdict(lambda: defaultdict(int))
    untokenized_possible_translations = defaultdict(
        lambda: defaultdict(int)
    )  # only filled when ALIGNMENT_UNIT is 'WORDS', and for diagnostics purposes only

    def add_token_pair(count, new_token, old_token):
        tokenized_possible_translations[new_token][old_token] += count

    def add_word_pair(count, new_word, old_word, all_to_all_mapping=False):
        # tokenize the words
        # (recall that we use the long hyphen to replace spaces inside words, to merge the tokens again)
        old_word_tokenized = old_word.split('—')
        new_word_tokenized = new_word.split('—')

        # if the token list dont have the same length, compute the smallest common multiple of their lengths
        if all_to_all_mapping:
            count_dilution = len(old_word_tokenized)
            old_word_tokenized = np.tile(old_word_tokenized, len(new_word_tokenized))
            new_word_tokenized = np.repeat(new_word_tokenized, count_dilution)
        elif len(old_word_tokenized) != len(new_word_tokenized):
            gcd = math.gcd(len(old_word_tokenized), len(new_word_tokenized))
            count_dilution = len(old_word_tokenized) // gcd
            old_word_tokenized = np.repeat(old_word_tokenized, len(new_word_tokenized) // gcd)
            new_word_tokenized = np.repeat(new_word_tokenized, count_dilution)
        else:
            gcd = 1
            count_dilution = 1

        # perform this operation for each token pair in the word
        for token_old, token_new in zip(old_word_tokenized, new_word_tokenized):
            # add the translation to the dictionary
            tokenized_possible_translations[token_new][token_old] += max(1, count // count_dilution)

    # determine the tokenizer settings
    OLD_TOKENIZER_1ST_PREFIX = old_tokenizer.convert_ids_to_tokens(
        old_tokenizer.encode(" a", add_special_tokens=False)[0]
    ).rstrip("a")
    NEW_TOKENIZER_1ST_PREFIX = new_tokenizer.convert_ids_to_tokens(
        new_tokenizer.encode(" a", add_special_tokens=False)[0]
    ).rstrip("a")
    OLD_TOKENIZER_2ND_PREFIX = old_tokenizer.convert_ids_to_tokens(
        old_tokenizer.encode("aaaaaaaaaaaaaaaaaaaaaa", add_special_tokens=False)[1]
    ).rstrip('a')
    NEW_TOKENIZER_2ND_PREFIX = new_tokenizer.convert_ids_to_tokens(
        new_tokenizer.encode("aaaaaaaaaaaaaaaaaaaaaa", add_special_tokens=False)[1]
    ).rstrip('a')

    total_alignments = 0
    with open(mapped_tokens_file) as f:
        for line in f:
            total_alignments += 1

    with open(mapped_tokens_file) as f:
        for line in tqdm(f, total=total_alignments):
            # remove the newline character
            line = line.rstrip('\n')
            # skip empty lines
            if line == '':
                continue
            # split the line on the tab character
            old_word, new_word, log_prob, count = line.split('\t')
            # reject <eps> mappings
            if old_word == '<eps>':
                continue
            if new_word == '<eps>':
                continue
            # convert the count to an integer
            count = int(float(count))
            # skip pairs that happened rarely (likely noise)
            if count < MIN_COUNT_REQUIRED_FOR_CONSIDERATION:
                continue
            # add the token pair to the token dictionary
            if (ALIGNMENT_UNIT != 'WORDS') or ((new_word in new_tokenizer_vocab) and (old_word in old_tokenizer_vocab)):
                add_token_pair(count, new_word, old_word)
            else:
                half_count = max(1, count // 2)
                add_word_pair(half_count, new_word, old_word, all_to_all_mapping=True)
                add_word_pair(half_count, new_word, old_word, all_to_all_mapping=False)
            # add the word translation to the dictionary (for diagnostics purposes only)
            untokenized_possible_translations[new_word][old_word] += count

    # add a mapping for all numbers
    for i in range(9999):
        str_i = str(i)
        if str_i in new_tokenizer_vocab:
            add_token_pair(1, str_i, str_i if str_i in old_tokenizer_vocab else old_tokenizer.tokenize(str_i)[0])
        if len(new_tokenizer.tokenize(str_i)) == 1:
            add_token_pair(1, new_tokenizer.tokenize(str_i)[0], old_tokenizer.tokenize(str_i)[0])
        if len(new_tokenizer.tokenize(' ' + str_i)) == 1:
            add_token_pair(1, new_tokenizer.tokenize(' ' + str_i)[0], old_tokenizer.tokenize(' ' + str_i)[0])
    for i in range(99):
        str_i = '0' + str(i)
        if str_i in new_tokenizer_vocab:
            add_token_pair(1, str_i, str_i if str_i in old_tokenizer_vocab else old_tokenizer.tokenize(str_i)[0])
        if len(new_tokenizer.tokenize(str_i)) == 1:
            add_token_pair(1, new_tokenizer.tokenize(str_i)[0], old_tokenizer.tokenize(str_i)[0])
        if len(new_tokenizer.tokenize(' ' + str_i)) == 1:
            add_token_pair(1, new_tokenizer.tokenize(' ' + str_i)[0], old_tokenizer.tokenize(' ' + str_i)[0])

    # add a mapping for all punctuation (and words that exist in both models)
    for token in new_tokenizer_vocab:
        ## skip if any token char is a letter or digit
        # if any(c.isalnum() for c in token): continue
        # replace the start symbol of the new model with the one of the old model
        if NEW_TOKENIZER_1ST_PREFIX != '' or OLD_TOKENIZER_1ST_PREFIX != '':
            token2 = token.replace(NEW_TOKENIZER_1ST_PREFIX, OLD_TOKENIZER_1ST_PREFIX)
        # replace the continuation symbol of the new model with the one of the old model
        if NEW_TOKENIZER_2ND_PREFIX != '' or OLD_TOKENIZER_2ND_PREFIX != '':
            token2 = token2.replace(NEW_TOKENIZER_2ND_PREFIX, OLD_TOKENIZER_2ND_PREFIX)
        # skip if token is not in the old model
        if token2 not in old_tokenizer_vocab:
            continue
        # add the mapping
        tokenized_possible_translations[token][token2] += 1

    def or_old_unk_token(token, fallback_token=None):
        if (token != None) and (token in old_tokenizer_vocab):
            return token
        if (fallback_token != None) and (fallback_token in old_tokenizer_vocab):
            return fallback_token
        return old_tokenizer.unk_token

    # add a mapping for special tokens (i.e. pad, unk, bos, eos, sep, cls, mask)
    very_large_number = 1_000_000_000
    if new_tokenizer.pad_token != None:
        add_token_pair(very_large_number, new_tokenizer.pad_token, or_old_unk_token(old_tokenizer.pad_token))
    if new_tokenizer.unk_token != None:
        add_token_pair(very_large_number, new_tokenizer.unk_token, or_old_unk_token(old_tokenizer.unk_token))
    if new_tokenizer.bos_token != None:
        add_token_pair(
            very_large_number,
            new_tokenizer.bos_token,
            or_old_unk_token(old_tokenizer.bos_token, old_tokenizer.cls_token),
        )
    if new_tokenizer.eos_token != None:
        add_token_pair(
            very_large_number,
            new_tokenizer.eos_token,
            or_old_unk_token(old_tokenizer.eos_token, old_tokenizer.sep_token),
        )
    if new_tokenizer.cls_token != None:
        add_token_pair(
            very_large_number,
            new_tokenizer.cls_token,
            or_old_unk_token(old_tokenizer.cls_token, old_tokenizer.bos_token),
        )
    if new_tokenizer.sep_token != None:
        add_token_pair(
            very_large_number,
            new_tokenizer.sep_token,
            or_old_unk_token(old_tokenizer.sep_token, old_tokenizer.eos_token),
        )
    if new_tokenizer.mask_token != None:
        add_token_pair(
            very_large_number,
            new_tokenizer.mask_token,
            or_old_unk_token(old_tokenizer.mask_token, old_tokenizer.pad_token),
        )

    # check how many tokens have a translation, compared to the total number of tokens
    print(f'Number of tokens with a translation: {len(tokenized_possible_translations)}')
    print(f'Number of new tokens: {len(new_tokenizer)}')
    print(f'Percentage of tokens with a translation: {int(len(tokenized_possible_translations) / len(new_tokenizer) * 1000)/10}%')

    return tokenized_possible_translations, untokenized_possible_translations

def smooth_mapping(target_tokenizer: str, tokenized_possible_translations: dict) -> dict:
        
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(target_tokenizer)

    # print the first 100 tokens that have no translation
    tmp_count = 0
    for i, token in enumerate(new_tokenizer.get_vocab()):
        #if tmp_count >= 100: break
        if token not in tokenized_possible_translations:
            tmp_count += 1
            # provide a list of tokens which start with the same characters
            similar_tokens = [token2 for token2 in new_tokenizer.get_vocab() if token2.startswith(token) and (token2 in tokenized_possible_translations)]
            ## find the tokens which are the start of this token
            #start_subset_tokens = [token2 for token2 in tokenized_possible_translations if token.startswith(token2) and (token2 in tokenized_possible_translations)]
            #start_subset_tokens.sort(key=lambda x: len(x), reverse=True)
            ## find the tokens which are the end of this token
            #end_subset_tokens = [token2 for token2 in tokenized_possible_translations if token.endswith(token2) and (token2 in tokenized_possible_translations)]
            #end_subset_tokens.sort(key=lambda x: len(x), reverse=True)
            # find the tokens which are the middle of this token
            middle_subset_tokens = [token2 for token2 in tokenized_possible_translations if (token2 in token) and (token2 in tokenized_possible_translations)]
            middle_subset_tokens.sort(key=lambda x: len(x), reverse=True)
            # remove the tokens which are included in another previous token of the list
            #start_subset_tokens = [token2 for i, token2 in enumerate(start_subset_tokens) if (i == 0) or not any([token2 in token3 for token3 in start_subset_tokens[0:i]])]
            #end_subset_tokens = [token2 for i, token2 in enumerate(end_subset_tokens) if (i == 0) or not any([token2 in token3 for token3 in end_subset_tokens[0:i]])]
            middle_subset_tokens = [token2 for i, token2 in enumerate(middle_subset_tokens) if (i == 0) or not any([token2 in token3 for token3 in middle_subset_tokens[0:i]])]
            # sort the middle tokens by position in the token
            middle_subset_tokens.sort(key=lambda x: token.index(x))
            # print the token, the similar tokens, and the start, end, and middle subset tokens
            if tmp_count <= 100: print(token, similar_tokens, middle_subset_tokens) #start_subset_tokens[0:3], end_subset_tokens[0:3], middle_subset_tokens[0:3])
            # add the token to the updated dictionary
            if len(similar_tokens) == 0 and len(middle_subset_tokens) == 0: continue
            tokenized_possible_translations[token] = defaultdict(int)
            for token2 in similar_tokens + middle_subset_tokens:
                # add all their translation to the dictionary, normalizing to a total count of 1000 for each token2 (2000 if the token starts with a space)
                count_for_token2 = sum(tokenized_possible_translations[token2].values())
                if count_for_token2 > 0:
                    for translation_of_token2 in tokenized_possible_translations[token2]:
                        weight = 2000 if translation_of_token2.startswith('▁') else 1000 # TODO: explain these values or parameterize them
                        tokenized_possible_translations[token][translation_of_token2] += max(1, (weight * tokenized_possible_translations[token2][translation_of_token2]) // count_for_token2)

    def get_coefficients_for_token(new_token):
        # check for unmapped tokens
        if new_token not in tokenized_possible_translations: return [(new_tokenizer.unk_token, 1.0)]
        # get the possible translations for this token
        possible_translations = tokenized_possible_translations[new_token]
        # get the total count of all translations
        total_count = sum(possible_translations.values())
        # check for unmapped tokens by count
        if total_count <= 0: return [(new_tokenizer.unk_token, 1.0)]
        # compute the probability of each translation
        probabilities = {old_token: count / total_count for old_token, count in possible_translations.items()}
        # sort the translations by probability
        probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        # return the probabilities
        return probabilities

    # convert the dictionary to a list of sorted lists, and save it to a json file
    final_list = []
    for token_i in range(len(new_tokenizer.vocab)):
        token = new_tokenizer.convert_ids_to_tokens(token_i)
        coefficients = get_coefficients_for_token(token)
        final_list.append((token, coefficients))

    return final_list

def remap_model(source_tokenizer: str, target_tokenizer: str, mapping: list[Tuple[str, list[Tuple[str, float]]]],  source_model: str) -> AutoModel:
    # load tokenizers for the two models
    old_tokenizer = transformers.AutoTokenizer.from_pretrained(source_tokenizer)
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(target_tokenizer)

    # load the old model
    model = transformers.AutoModelForCausalLM.from_pretrained(source_model)

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

    os.makedirs('output', exist_ok=False)
    model.save_pretrained('output/')
    new_tokenizer.save_pretrained('output/')

    return model

if __name__ == "__main__":
    source_tokenizer= "mistralai/Mistral-7B-v0.1"
    target_tokenizer= "dtai-kuleuven/robbert-2023-dutch-base"

    corpus = create_aligned_corpus(
        source_language="en",
        target_language="nl",
        source_tokenizer="mistralai/Mistral-7B-v0.1",
        target_tokenizer=target_tokenizer,
    )

    mapped_tokens_file = align(
        corpus, fast_align_path="/Users/pieter/Documents/2024/tik-to-tok/notebooks/fast_align/build/fast_align"
    )

    tokenized_possible_translations, untokenized_possible_translations = map_tokens(
        mapped_tokens_file, "mistralai/Mistral-7B-v0.1", "dtai-kuleuven/robbert-2023-dutch-base"
    )

    smoothed_mapping = smooth_mapping(target_tokenizer, tokenized_possible_translations)

    remap_model(source_tokenizer, target_tokenizer, smoothed_mapping, source_tokenizer)