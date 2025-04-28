import os
import platform
import sys
import psutil
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, get_compute_capability
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.vendors import device_manager, GPUVendor

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

def load_model(
    model_path,
    optimize_config_path,
    gguf_path,
    cpu_infer=10,
    use_cuda_graph=True,
    mode="normal",
    force_think=False,
    chunk_size=8192,
    use_swapper=False
):
    torch.set_grad_enabled(False)
    Config().cpu_infer = cpu_infer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if "Qwen2Moe" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"generation config can't auto create, make default. Message: {e}")
        gen_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        model.generation_config = gen_config
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    print("模型加载后内存占用 (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    return model, tokenizer, config

def run_inference(
    model, tokenizer, config, prompt,
    max_new_tokens=1000,
    use_cuda_graph=True,
    mode="normal",
    force_think=False,
    chunk_size=8192
):
    messages = [{"role": "user", "content": prompt}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if force_think:
        token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
        input_tensor = torch.cat(
            [input_tensor, token_thinks], dim=1
        )
    if mode == 'long_context':
        assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
        "please change max_seq_len in  ~/.ktransformers/config.yaml"

    system = platform.system()
    if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8 and device_manager.gpu_vendor == GPUVendor.NVIDIA:
        generated = prefill_and_generate(
            model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
            use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
        )
    else:
        generated = prefill_and_generate(
            model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_size = chunk_size,
        )
    print("推理后内存占用 (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    return generated