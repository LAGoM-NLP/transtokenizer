# tik-to-tok


[![pypi](https://img.shields.io/pypi/v/tik-to-tok.svg)](https://pypi.org/project/tik-to-tok/)
[![python](https://img.shields.io/pypi/pyversions/tik-to-tok.svg)](https://pypi.org/project/tik-to-tok/)
[![Build Status](https://github.com/ipieter/tik-to-tok/actions/workflows/dev.yml/badge.svg)](https://github.com/ipieter/tik-to-tok/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/ipieter/tik-to-tok/branch/main/graphs/badge.svg)](https://codecov.io/github/ipieter/tik-to-tok)



Token translation for language models


* Documentation: <https://ipieter.github.io/tik-to-tok>
* GitHub: <https://github.com/ipieter/tik-to-tok>
* PyPI: <https://pypi.org/project/tik-to-tok/>
* Licence: MIT


## Features

* TODO

## Usage

```python
from tiktotok import transform_model
from transformers import AutoTokenizer, AutoModelForCausalLM

source_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
target_tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-2023-dutch-base")

source_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

target_model = transform_model(source_model, source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer)
```

## Credits
