import fire
import psutil
import os
#from ktransformers.local_chat import local_chat
from ktransformers.local_chat import load_model,run_inference
# 你要批量测试的prompt列表
prompts = [
    "你好",
    "请写一个快速排序的Python代码。",
    "什么是Transformer模型？",
    "请用C++实现二分查找。",
    "人工智能的发展历史是什么？",
    "请写一首关于春天的诗。",
    "介绍一下深度学习和机器学习的区别。",
    "请用Java实现单例模式。",
    "什么是注意力机制？",
    "请用英文自我介绍。",
]

def batch_test(
    model_path,
    optimize_config_path,
    gguf_path,
    max_new_tokens=100,
    cpu_infer=10,
    use_cuda_graph=True,
    mode="normal",
    use_swapper=False,
    chunk_size=8192
):
    # 初始化模型，只加载一次
    def single_chat(prompt):
        # 这里直接调用local_chat的主要推理部分
        # 可以把local_chat的推理部分抽成一个函数，然后这里调用
        # 或者直接用os.system调用命令行（见下方注释）
        pass
    
    # 只加载一次模型
    model, tokenizer, config = load_model(
        model_path=model_path,
        optimize_config_path=optimize_config_path,
        gguf_path=gguf_path,
        cpu_infer=cpu_infer,
        use_swapper=use_swapper,
        use_cuda_graph=use_cuda_graph,
        mode=mode,
        chunk_size=chunk_size
    )
    # 或者直接用命令行方式批量调用
    for prompt in prompts:
        print("="*40)
        #print("Prompt:", prompt)
        # 直接用os.system调用local_chat.py
        output = run_inference(model, tokenizer, config, prompt, max_new_tokens=max_new_tokens)
        # 写入prompt到临时文件
        #with open("/tmp/prompt.txt", "w") as f:
         #   f.write(prompt)
        #os.system(cmd)
        # 记录内存
        #print("Output:",output)
        print("当前内存占用 (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)

if __name__ == "__main__":
    fire.Fire(batch_test)