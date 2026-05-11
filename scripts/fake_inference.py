# Copyright (c) 2025, X-Square-Robot
# SPDX-License-Identifier: BSD-3-Clause

import torch
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction

model_path = "/path/to/model"
model = Qwen2_5_VLMoEForAction.from_pretrained(model_path)
model.eval()

# Gen Fake data
batch_size = 1
seq_length = 50

torch.manual_seed(0)
fake_input_ids = torch.randint(
    0, len(model.processor.tokenizer), (batch_size, seq_length), dtype=torch.long
)
fake_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
fake_moe_token_types = torch.zeros((batch_size, seq_length), dtype=torch.long)
fake_position_ids = (
    torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
)
fake_proprioception = torch.randn((batch_size, 1, 20), dtype=torch.float32)
fake_agent_pos_mask = torch.ones((batch_size, 1, 20), dtype=torch.float32)
fake_dof_mask = torch.ones((batch_size, 32, 20), dtype=torch.float32)
fake_dataset_names = ["x2_normal"]


device = "cuda"

model = model.to(device)
model = model.bfloat16()

fake_input_ids = fake_input_ids.to(device)
fake_attention_mask = fake_attention_mask.to(device)
fake_moe_token_types = fake_moe_token_types.to(device)
fake_position_ids = fake_position_ids.to(device)
fake_proprioception = fake_proprioception.to(device).bfloat16()
fake_agent_pos_mask = fake_agent_pos_mask.to(device).bfloat16()
fake_dof_mask = fake_dof_mask.to(device).bfloat16()

try:
    with torch.no_grad():
        outputs = model(
            input_ids=fake_input_ids,
            attention_mask=fake_attention_mask,
            moe_token_types=fake_moe_token_types,
            position_ids=fake_position_ids,
            proprioception=fake_proprioception,
            agent_pos_mask=fake_agent_pos_mask,
            dof_mask=fake_dof_mask,
            dataset_names=fake_dataset_names,
            mode="validate",
        )

    print("✅ Fake inference test successful!")
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output logits dtype: {outputs.logits.dtype}")
    print(f"Output logits device: {outputs.logits.device}")

    # Check if output is reasonable
    if outputs.logits.shape == (batch_size, seq_length, model.config.vocab_size):
        print("✅ Output shape correct")
    else:
        print("❌ Output shape incorrect")

    if not torch.isnan(outputs.logits).any():
        print("✅ Output contains no NaN values")
    else:
        print("❌ Output contains NaN values")

    if not torch.isinf(outputs.logits).any():
        print("✅ Output contains no infinity values")
    else:
        print("❌ Output contains infinity values")

    print("Output logits statistics:")
    print(f"  Min value: {outputs.logits.min().item():.4f}")
    print(f"  Max value: {outputs.logits.max().item():.4f}")
    print(f"  Mean: {outputs.logits.mean().item():.4f}")
    print(f"  Standard deviation: {outputs.logits.std().item():.4f}")

except Exception as e:
    print(f"❌ Fake inference test failed: {e}")
    import traceback

    traceback.print_exc()
