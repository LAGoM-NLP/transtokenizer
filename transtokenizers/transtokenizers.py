
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
MIN_COUNT_REQUIRED_FOR_CONSIDERATION = 10

home_path = os.environ['TT_HOME'] if "TT_HOME" in os.environ else Path("export")

CCMATRIX_MAPPING = {'af': 'afr_Latn',
 'sq': 'als_Latn',
 'am': 'amh_Ethi',
 'ar': 'arb_Arab',
 'ast': 'ast_Latn',
 'az': 'azj_Latn',
 'be': 'bel_Cyrl',
 'bn': 'ben_Beng',
 'bg': 'bul_Cyrl',
 'ca': 'cat_Latn',
 'ceb': 'ceb_Latn',
 'cs': 'ces_Latn',
 'cy': 'cym_Latn',
 'da': 'dan_Latn',
 'de': 'deu_Latn',
 'el': 'ell_Grek',
 'en': 'eng_Latn',
 'eo': 'epo_Latn',
 'et': 'est_Latn',
 'fi': 'fin_Latn',
 'fr': 'fra_Latn',
 'om': 'gaz_Latn',
 'gd': 'gla_Latn',
 'ga': 'gle_Latn',
 'gl': 'glg_Latn',
 'ha': 'hau_Latn',
 'he': 'heb_Hebr',
 'hi': 'hin_Deva',
 'hr': 'hrv_Latn',
 'hu': 'hun_Latn',
 'hy': 'hye_Armn',
 'ig': 'ibo_Latn',
 'ilo': 'ilo_Latn',
 'id': 'ind_Latn',
 'is': 'isl_Latn',
 'it': 'ita_Latn',
 'jv': 'jav_Latn',
 'ja': 'jpn_Jpan',
 'ka': 'kat_Geor',
 'kk': 'kaz_Cyrl',
 'km': 'khm_Khmr',
 'ko': 'kor_Hang',
 'lt': 'lit_Latn',
 'lb': 'ltz_Latn',
 'lg': 'lug_Latn',
 'lv': 'lvs_Latn',
 'ml': 'mal_Mlym',
 'mr': 'mar_Deva',
 'mk': 'mkd_Cyrl',
 'my': 'mya_Mymr',
 'nl': 'nld_Latn',
 'no': 'nob_Latn',
 'ne': 'npi_Deva',
 'oc': 'oci_Latn',
 'or': 'ory_Orya',
 'fa': 'pes_Arab',
 'mg': 'plt_Latn',
 'pl': 'pol_Latn',
 'pt': 'por_Latn',
 'ro': 'ron_Latn',
 'ru': 'rus_Cyrl',
 'si': 'sin_Sinh',
 'sk': 'slk_Latn',
 'sl': 'slv_Latn',
 'sd': 'snd_Arab',
 'so': 'som_Latn',
 'es': 'spa_Latn',
 'sr': 'srp_Cyrl',
 'su': 'sun_Latn',
 'sv': 'swe_Latn',
 'sw': 'swh_Latn',
 'ta': 'tam_Taml',
 'tt': 'tat_Cyrl',
 'tl': 'tgl_Latn',
 'tr': 'tur_Latn',
 'uk': 'ukr_Cyrl',
 'ur': 'urd_Arab',
 'uz': 'uzn_Latn',
 'vi': 'vie_Latn',
 'wo': 'wol_Latn',
 'xh': 'xho_Latn',
 'yi': 'ydd_Hebr',
 'yo': 'yor_Latn',
 'zh': 'zho_Hans',
 'ms': 'zsm_Latn',
 'zu': 'zul_Latn'} # from NLLB repo: https://huggingface.co/datasets/allenai/nllb/tree/main

DEFAUT_SCRIPT_BY_LANG = {'ace': 'Latn', 'ban': 'Latn', 'bjn': 'Latn', 'bug': 'Latn', 'ceb': 'Latn', 'eng': 'Latn', 'fij': 'Latn', 'ilo': 'Latn', 'jav': 'Latn', 'min': 'Latn', 'mri': 'Latn', 'pag': 'Latn', 'plt': 'Latn', 'smo': 'Latn', 'sun': 'Latn', 'war': 'Latn', 'afr': 'Latn', 'aka': 'Latn', 'amh': 'Ethi', 'bam': 'Latn', 'bem': 'Latn', 'cjk': 'Latn', 'dik': 'Latn', 'dyu': 'Latn', 'ewe': 'Latn', 'fon': 'Latn', 'fra': 'Latn', 'fuv': 'Latn', 'gaz': 'Latn', 'hau': 'Latn', 'ibo': 'Latn', 'kam': 'Latn', 'kik': 'Latn', 'kin': 'Latn', 'kmb': 'Latn', 'knc': 'Arab', 'kon': 'Latn', 'lin': 'Latn', 'lua': 'Latn', 'lug': 'Latn', 'luo': 'Latn', 'nso': 'Latn', 'nus': 'Latn', 'nya': 'Latn', 'run': 'Latn', 'sna': 'Latn', 'som': 'Latn', 'sot': 'Latn', 'ssw': 'Latn', 'swh': 'Latn', 'tir': 'Ethi', 'tsn': 'Latn', 'tso': 'Latn', 'tum': 'Latn', 'twi': 'Latn', 'umb': 'Latn', 'wol': 'Latn', 'xho': 'Latn', 'yor': 'Latn', 'zul': 'Latn', 'arb': 'Arab', 'ckb': 'Arab', 'crh': 'Latn', 'diq': 'Latn', 'kmr': 'Latn', 'tat': 'Cyrl', 'tzm': 'Tfng', 'urd': 'Arab', 'asm': 'Beng', 'awa': 'Deva', 'ben': 'Beng', 'bho': 'Deva', 'guj': 'Gujr', 'hin': 'Deva', 'hne': 'Deva', 'kan': 'Knda', 'kas': 'Arab', 'mag': 'Deva', 'mai': 'Deva', 'mal': 'Mlym', 'mar': 'Deva', 'npi': 'Deva', 'ory': 'Orya', 'pan': 'Guru', 'san': 'Deva', 'sat': 'Beng', 'sin': 'Sinh', 'snd': 'Arab', 'tam': 'Taml', 'tel': 'Telu', 'ayr': 'Latn', 'spa': 'Latn', 'azb': 'Arab', 'azj': 'Latn', 'rus': 'Cyrl', 'bak': 'Cyrl', 'kir': 'Cyrl', 'tuk': 'Latn', 'uig': 'Arab', 'uzn': 'Latn', 'bel': 'Cyrl', 'pbt': 'Arab', 'ind': 'Latn', 'bod': 'Tibt', 'bos': 'Latn', 'por': 'Latn', 'prs': 'Arab', 'tgk': 'Cyrl', 'cym': 'Latn', 'dzo': 'Tibt', 'als': 'Latn', 'epo': 'Latn', 'fao': 'Latn', 'fur': 'Latn', 'gla': 'Latn', 'gle': 'Latn', 'grn': 'Latn', 'hat': 'Latn', 'hye': 'Armn', 'kab': 'Latn', 'kac': 'Latn', 'kat': 'Geor', 'kaz': 'Cyrl', 'kbp': 'Latn', 'kea': 'Latn', 'khk': 'Cyrl', 'khm': 'Khmr', 'lao': 'Laoo', 'lij': 'Latn', 'lim': 'Latn', 'lmo': 'Latn', 'ltg': 'Latn', 'ltz': 'Latn', 'lus': 'Latn', 'mlt': 'Latn', 'mni': 'Beng', 'mos': 'Latn', 'mya': 'Mymr', 'pap': 'Latn', 'quy': 'Latn', 'sag': 'Latn', 'scn': 'Latn', 'shn': 'Mymr', 'srd': 'Latn', 'szl': 'Latn', 'taq': 'Latn', 'tgl': 'Latn', 'tpi': 'Latn', 'vec': 'Latn', 'ydd': 'Hebr', 'zho': 'Hans', 'zsm': 'Latn', 'glg': 'Latn', 'oci': 'Latn', 'dan': 'Latn', 'deu': 'Latn', 'isl': 'Latn', 'nld': 'Latn', 'nob': 'Latn', 'swe': 'Latn', 'tur': 'Latn', 'srp': 'Cyrl', 'ukr': 'Cyrl', 'bul': 'Cyrl', 'cat': 'Latn', 'ces': 'Latn', 'ell': 'Grek', 'est': 'Latn', 'fin': 'Latn', 'heb': 'Hebr', 'hrv': 'Latn', 'hun': 'Latn', 'ita': 'Latn', 'jpn': 'Jpan', 'kor': 'Hang', 'lit': 'Latn', 'lvs': 'Latn', 'pes': 'Arab', 'pol': 'Latn', 'ron': 'Latn', 'slk': 'Latn', 'slv': 'Latn', 'vie': 'Latn', 'ast': 'Latn', 'mkd': 'Cyrl'}

def get_dataset_iterator(dataset_name: str, source_language: str, target_language: str):
    """Utility function to get the iterable of a dataset, mapping different dataset conventions.

    Currently supported:
    - open_subtitles
    - allenai/nllb
    """

    if dataset_name == "open_subtitles":
        # convert the langauge to a 2-letter code, if needed
        source_language = Language.get(source_language).language if len(source_language) != 2 else source_language
        target_language = Language.get(target_language).language if len(target_language) != 2 else target_language

        # load the dataset
        dataset = load_dataset(dataset_name, lang1=source_language, lang2=target_language, streaming=True, trust_remote_code=True)

        # wrap the dataset iterator so it returns a tuple of the source and target sentences
        class DatasetWrapper(Iterator):
            def __init__(self, dataset):
                self.dataset = iter(dataset['train'])

            def __next__(self):
                example = next(self.dataset)
                return example['translation'][source_language], example['translation'][target_language]

    elif dataset_name == "allenai/nllb":
        # map 2-letter language codes to 3-letter language codes (ISO 639-2 Code from ISO 639-1 Code)
        src_lang = Language.get(source_language)
        tgt_lang = Language.get(target_language)
        src_script = src_lang.script if src_lang.script else DEFAUT_SCRIPT_BY_LANG[src_lang.to_alpha3()]
        tgt_script = tgt_lang.script if tgt_lang.script else DEFAUT_SCRIPT_BY_LANG[tgt_lang.to_alpha3()]

        # load the dataset
        dataset = load_dataset(
            dataset_name,
            f"{src_lang.to_alpha3()}_{src_script}-{tgt_lang.to_alpha3()}_{tgt_script}",
            split='train',
            streaming=True,
            trust_remote_code=True
        )

        # wrap the dataset iterator so it returns a tuple of the source and target sentences
        class DatasetWrapper(Iterator):
            def __init__(self, dataset):
                self.dataset = iter(dataset)

            def __next__(self):
                example = next(self.dataset)
                src_lang = Language.get(source_language)
                tgt_lang = Language.get(target_language)
                src_script = src_lang.script if src_lang.script else DEFAUT_SCRIPT_BY_LANG[src_lang.to_alpha3()]
                tgt_script = tgt_lang.script if tgt_lang.script else DEFAUT_SCRIPT_BY_LANG[tgt_lang.to_alpha3()]
                return (
                    example['translation'][f"{src_lang.to_alpha3()}_{src_script}"],
                    example['translation'][f"{tgt_lang.to_alpha3()}_{tgt_script}"],
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

    output_path = corpus.replace(".moses", ".fast_align.tsv")

    if os.path.exists(output_path):

        print(f"corpus already aligned")

    else:

        # check if fast_align is installed
        if os.system(f"{fast_align_path} -h") != 256:
            raise ValueError(
                "fast_align is not installed. Please install it from https://github.com/FremyCompany/fast_align"
            )

        # call fast_align
        os.system(f'{fast_align_path} -i {corpus} -I 7 -p {output_path} > /dev/null')

    return output_path

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

def smooth_mapping(target_tokenizer: str, tokenized_possible_translations: dict, print_debug=False) -> dict:
    """
    Tokens (e.g. long words) that don't get mapped from the parallel corpus are mapped using split tokens. All tokens inside that token 
    and don't overlap, are used for that mapping. 

    TODO example
    """
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(target_tokenizer)

    # print the first 100 tokens that have no translation
    tmp_count = 0
    for i, token in enumerate(tqdm(new_tokenizer.get_vocab())):
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
            if print_debug and tmp_count <= 100: print(token, similar_tokens, middle_subset_tokens) #start_subset_tokens[0:3], end_subset_tokens[0:3], middle_subset_tokens[0:3])
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

    # add an unk token if none exist
    if old_tokenizer.unk_token_id is None:
        if not(old_tokenizer.pad_token_id is None):
            old_tokenizer.unk_token_id = old_tokenizer.pad_token_id
            old_tokenizer.unk_token = old_tokenizer.pad_token
        elif not(old_tokenizer.bos_token_id is None):
            old_tokenizer.unk_token_id = old_tokenizer.bos_token_id
            old_tokenizer.unk_token = old_tokenizer.bos_token
        else:
            print("WARNING: The old tokenizer had neither UNK, PAD or BOS special tokens")
            old_tokenizer.unk_token_id = 0

    # load the old model
    print("Loading the source model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(source_model)

    # remap the embeddings
    print("Remapping the model...")
    with torch.no_grad():
        # get the embeddings of the OLM model
        old_embeddings = model.get_input_embeddings().weight.data
        old_output_embeddings = model.get_output_embeddings().weight.data
        tied_weights = model.config.tie_word_embeddings

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

        # debug info
        #print(old_embeddings.shape)
        #print(old_output_embeddings.shape)
        #print(new_embeddings.weight.data.shape)
        #print(new_output_embeddings.weight.data.shape)

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
            old_ids = [(old_id if not(old_id is None) else old_tokenizer.unk_token_id, weight) for old_id, weight in old_ids]

            # we use a weighted average, where the first token in the list has 0.4 weight, the second 0.2, and the remaining 0.4 are equally distributed among all tokens (including the first two)
            if len(old_ids) > 1:
                new_embeddings.weight.data[new_id] = reduce(lambda a, b: a.add_(b), [old_embeddings[old_id]*weight for old_id, weight in old_ids])
                if not(tied_weights): new_output_embeddings.weight.data[new_id] = reduce(lambda a, b: a.add_(b), [old_output_embeddings[old_id]*weight for old_id, weight in old_ids])
            elif len(old_ids) == 1:
                new_embeddings.weight.data[new_id] = old_embeddings[old_ids[0][0]]
                if not(tied_weights): new_output_embeddings.weight.data[new_id] = old_output_embeddings[old_ids[0][0]]
            else: # use the unknown token embedding if the token is not in the old tokenizer
                new_embeddings.weight.data[new_id] = old_embeddings[old_tokenizer.unk_token_id]
                if not(tied_weights): new_output_embeddings.weight.data[new_id] = old_output_embeddings[old_tokenizer.unk_token_id]

    return model

if __name__ == "__main__":
    # TODO: transform into an Argument Parser!
    source_language  = "en"
    source_tokenizer = "mistralai/Mistral-7B-v0.1"
    target_language  = "nl"
    target_tokenizer = "dtai-kuleuven/robbert-2023-dutch-base"
    corpus_list      = ["open_subtitles", "allenai/nllb"]
    fast_align_path  = f"{home_path}/notebooks/fast_align/build/fast_align"

    corpus = create_aligned_corpus(
        source_language=source_language,
        target_language=target_language,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        corpus_list=corpus_list
    )

    mapped_tokens_file = align(
        corpus, fast_align_path=fast_align_path
    )

    tokenized_possible_translations, untokenized_possible_translations = map_tokens(
        mapped_tokens_file, source_tokenizer, target_tokenizer
    )

    smoothed_mapping = smooth_mapping(target_tokenizer, tokenized_possible_translations)

    model = remap_model(source_tokenizer, target_tokenizer, smoothed_mapping, source_tokenizer)

    os.makedirs('output', exist_ok=False)
    model.save_pretrained('output/')
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(target_tokenizer)
    new_tokenizer.save_pretrained('output/')
