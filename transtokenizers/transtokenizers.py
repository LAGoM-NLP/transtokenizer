import os
from pathlib import Path
from typing import Iterator
import regex as re
from tqdm import tqdm
from langcodes import Language

import torch
import torch.nn as nn

import transformers
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

        if os.path.exists(
            out_path
        ):
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
        raise ValueError("fast_align is not installed. Please install it from https://github.com/FremyCompany/fast_align")

    # call fast_align
    os.system(f'{fast_align_path} -i {corpus} -I 7 -p {corpus.replace(".moses", ".fast_align.tsv")} > /dev/null')

    return corpus.replace(".moses", ".fast_align.tsv")


def map_tokens(mapped_tokens_file: str):
    pass

if __name__ == "__main__":
    corpus = create_aligned_corpus(
        source_language="en",
        target_language="nl",
        source_tokenizer="mistralai/Mistral-7B-v0.1",
        target_tokenizer="dtai-kuleuven/robbert-2023-dutch-base",
    )

    mapped_tokens_file = align(corpus, fast_align_path="/Users/pieter/Documents/2024/tik-to-tok/notebooks/fast_align/build/fast_align")


