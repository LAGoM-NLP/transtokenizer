import os
from transtokenizers.transtokenizers import align, create_aligned_corpus, smooth_mapping, map_tokens, remap_model
from transformers import AutoTokenizer
import datasets

if __name__ == "__main__":
    source_model = "meta-llama/Meta-Llama-3-8B"
    source_tokenizer = "meta-llama/Meta-Llama-3-70B"
    target_tokenizer = "yhavinga/gpt-neo-1.3B-dutch"
    export_dir = "en-nl-llama3-8b"

    corpus = create_aligned_corpus(
        source_language="en",
        target_language="nl",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    mapped_tokens_file = corpus.replace(".moses", ".fast_align.tsv")
    if not os.path.exists(mapped_tokens_file):
        align(corpus, fast_align_path="/cw/dtaijupiter/NoCsBack/dtai/pieterd/projects/tik-to-tok/fast_align/build/fast_align")

    tokenized_possible_translations, untokenized_possible_translations = map_tokens(mapped_tokens_file, source_tokenizer, target_tokenizer)

    smoothed_mapping = smooth_mapping(target_tokenizer, tokenized_possible_translations)

    model = remap_model(source_tokenizer, target_tokenizer, smoothed_mapping, source_tokenizer)
    os.makedirs(export_dir, exist_ok=False)
    new_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
    model.save_pretrained(export_dir)
    new_tokenizer.save_pretrained(export_dir)
