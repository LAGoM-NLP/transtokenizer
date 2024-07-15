import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM , AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import datasets


def calculate_perplexity(model, dataloader, device, tokenizer):
    model.eval()  # make sure the model is in evaluation mode
    total_loss = 0
    total_examples = 0
    non_padded_count = 0
    with torch.no_grad():  # we don't need to calculate gradients
        for batch in dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(input_ids=inputs, labels=labels, attention_mask=(inputs != tokenizer.pad_token_id).long())
            total_loss += outputs.loss.item() * inputs.size(0)
            total_examples += inputs.size(0)

            average_loss = total_loss / total_examples
            perplexity = torch.exp(torch.tensor(average_loss))

            non_padded_count += (inputs != tokenizer.pad_token_id).sum().item()

            tqdm.write(f"Perplexity: {perplexity:.2f}, count: {non_padded_count:.2f}")

    return perplexity, non_padded_count


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate the perplexity of a model on a dataset.')
    parser.add_argument('--model_name', type=str, help='Model name (e.g., "gpt2")', required=True)
    parser.add_argument('--dataset_name', type=str, help='Dataset name (e.g., "wikitext")', required=True)
    parser.add_argument('--subset', type=str, help='Possible subset of the dataset (e.g., "wikitext-2-raw-v1")')

    args = parser.parse_args()
    dataset = load_dataset(args.dataset_name, args.subset, split="validation", streaming=True) if args.subset else load_dataset(args.dataset_name, split="validation", streaming=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)  
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,  padding_side="right")

    # Assuming you have a GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch_size = 4

    def gen(dataset, tokenizer):
        for example in tqdm(dataset, dynamic_ncols=True):
            text = example['text']
            tokenized_text = tokenizer.encode(text, padding="max_length", return_tensors='pt', truncation=True, return_special_tokens_mask=True, max_length=1024 )[0]
            yield tokenized_text
    
    dataset = datasets.IterableDataset.from_generator(
        generator=gen,
        gen_kwargs={'dataset': dataset,
                    'tokenizer': tokenizer})

    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer has {len(tokenizer)} tokens")

    # Create the DataLoader for our test dataset
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)


    ppl, tokens = calculate_perplexity(model, test_dataloader, device, tokenizer)
    print(f"--------------{ppl=}-------------")
    print(f"--------------{tokens=}-------------")

