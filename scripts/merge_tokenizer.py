from transformers import AutoProcessor
import os

processor_path = "/path/to/Qwen2.5-VL-3B-Instruct"
action_tokenizer_path = "/path/to/fast"
use_fast_tokenizer = True

processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
processor.tokenizer.padding_side = "left"

action_tokenizer = AutoProcessor.from_pretrained(
    action_tokenizer_path, trust_remote_code=True
)

new_tokens = ["<|propri|>", "<|action|>"]
new_tokens += [f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)]
num_added_tokens = processor.tokenizer.add_tokens(new_tokens)

begin_idx_token = "<|action_token_0|>"
token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
processor.tokenizer.init_kwargs["action_token_vocab_size"] = action_tokenizer.vocab_size

new_tokenizer_dir = "/path/to/new_tokenizer"
os.makedirs(new_tokenizer_dir, exist_ok=True)
processor.save_pretrained(new_tokenizer_dir)
