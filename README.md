# transtokenizers


[![pypi](https://img.shields.io/pypi/v/trans-tokenizers.svg)](https://pypi.org/project/trans-tokenizers/)
[![python](https://img.shields.io/pypi/pyversions/trans-tokenizers.svg)](https://pypi.org/project/trans-tokenizers/)
[![Build Status](https://github.com/ipieter/transtokenizer/actions/workflows/dev.yml/badge.svg)](https://github.com/ipieter/transtokenizer/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/ipieter/transtokenizer/branch/main/graphs/badge.svg)](https://codecov.io/github/ipieter/transtokenizer)



Token translation for language models


* Documentation: <https://ipieter.github.io/transtokenizer>
* GitHub: <https://github.com/ipieter/transtokenizer>
* PyPI: <https://pypi.org/project/trans-tokenizers/>
* Licence: MIT


## Features

* TODO

## Usage

```python
from transtokenizers import transform_model
from transformers import AutoTokenizer, AutoModelForCausalLM

source_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
target_tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-2023-dutch-base")

source_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

target_model = transform_model(source_model, source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer)
```

## Creating a tokenizer
You might want to create a new tokenizer for the language that you are targetting. We provide a small script to create a SentencePiece tokenizer on the same datasets that you want to train on (e.g. NLLB). 
To create a tokenizer, you can use the following command:

```sh
python3 transtokenizers/tokenization.py --language als --dataset "allenai/nllb" --subset "eng_Latn-als_Latn"
```

This example generates a Albanian (sq, als) tokenizer from the NLLB dataset.
The final tokenizer is saved in the `tokenizers` directory under `tokenizers/als/`.

## Credits
If this repo was useful to you, please cite the following paper

```bibtex

```