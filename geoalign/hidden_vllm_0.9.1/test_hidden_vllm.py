import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams
import torch

llm = LLM(
    model="./Qwen3-4B-Base/Qwen/Qwen3-4B-Base",
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)


prompts = [
    "The capital of Japan is",
    "Explain LLM in simple terms.",
    "Explain gravity in simple terms.",
    "The capital of France is",
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=10,
    skip_special_tokens=True,
)

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"\n=== Prompt {i+1}: {prompts[i]} ===")
    print(f"Generated text: {output.outputs[0].text}")

    output = output.outputs[0]

    if hasattr(output, 'token_hiddens') and output.token_hiddens is not None:
        print(f"{len(output.token_ids)} token_ids: {output.token_ids}")
        print(f"token_hiddens: {len(output.token_hiddens)} sequences")
        print(f"first token hidden: {output.token_hiddens[0]}")
    else:
        print("hidden_states_decode not found or None")