# transtokenizers


[![pypi](https://img.shields.io/pypi/v/trans-tokenizers.svg)](https://pypi.org/project/trans-tokenizers/)
[![python](https://img.shields.io/pypi/pyversions/trans-tokenizers.svg)](https://pypi.org/project/trans-tokenizers/)

Token translation for language models


* GitHub: <https://github.com/LAGoM-NLP/transtokenizer>
* PyPI: <https://pypi.org/project/trans-tokenizers/>
* Licence: MIT


## Features

* Translate a model from one language to another.
* Support for most scripts beyond Latin.

## Installation

```bash
pip install trans-tokenizers
```


## Usage

You do need an installation of fast_align to align the tokens. You can install from the following repo: [https://github.com/FremyCompany/fast_align](https://github.com/FremyCompany/fast_align).

To convert a Llama model from English to Dutch, you can use the following code. This might 

```python
from transtokenizers import create_aligned_corpus, align, map_tokens, smooth_mapping, remap_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

source_model = "meta-llama/Meta-Llama-3-8B"

target_tokenizer = "yhavinga/gpt-neo-1.3B-dutch"
export_dir = "en-nl-llama3-8b"

corpus = create_aligned_corpus(
    source_language="en",
    target_language="nl",
    source_tokenizer=source_model,
    target_tokenizer=target_tokenizer,
)

mapped_tokens_file = align(corpus, fast_align_path="fast_align")

tokenized_possible_translations, untokenized_possible_translations = map_tokens(mapped_tokens_file, source_model, target_tokenizer)

smoothed_mapping = smooth_mapping(target_tokenizer, tokenized_possible_translations)

model = remap_model(source_model, target_tokenizer, smoothed_mapping, source_model)
os.makedirs(export_dir, exist_ok=False)
new_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
model.save_pretrained(export_dir)
new_tokenizer.save_pretrained(export_dir)
```
## 

## Credits
If this repo was useful to you, please cite the following paper

```bibtex
@inproceedings{remy-delobelle2024transtokenization,
    title={Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of {LLM}s for Low-Resource {NLP}},
    author={Remy, Fran{\c{c}}ois and Delobelle, Pieter and Avetisyan, Hayastan and Khabibullina, Alfiya and de Lhoneux, Miryam and Demeester, Thomas},
    booktitle={First Conference on Language Modeling},
    year={2024},
    url={https://openreview.net/forum?id=sBxvoDhvao}
}
```