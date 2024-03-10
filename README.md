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

## Credits
