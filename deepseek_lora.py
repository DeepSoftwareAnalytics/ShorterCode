import torch
import re
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
from datasets import DatasetDict

# 配置路径（根据实际路径修改）
model_path = r"/root/autodl-tmp/hyx/deepseek-coder-1.3b-base"  # 模型路径
data_path = r"/root/autodl-tmp/hyx/shortercode/dataset/dataset_all.jsonl"  # 数据集路径
output_path = r"./output/deepseek_lora"  # 微调后模型保存路径
# 强制使用GPU
assert torch.cuda.is_available()
device = torch.device("cuda")

class EvalLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f"[Eval] Step {state.global_step} - Eval Loss: {metrics['eval_loss']:.4f}")


# 自定义回调记录Loss
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])
# 数据预处理函数
# def process_data(tokenizer):
#     dataset = load_dataset("json", data_files=data_path, split="train[:996]")
#     def format_example(example):
#         # instruction = f"text：{example['instruction']}"
#         code=example['output']
#         # 优先匹配函数定义，其次匹配类定义
#         match = re.search(r'^(def\s+\w+\s*\(.*?\):)', code, re.MULTILINE)
#         if not match:
#             match = re.search(r'^(class\s+\w+\s*(\(.*?\))?:)', code, re.MULTILINE)

#         # 如果都没匹配上，就取第一行非空行
#         if match:
#             definition_line = match.group(1)
#         else:
#             lines = [line.strip() for line in code.splitlines() if line.strip()]
#             definition_line = lines[0] if lines else "# No code found"

#         # 拼接注释
#         prompt = f"{definition_line}  # {example['instruction']}"
#         print(prompt)
        
#         inputs = tokenizer(
#         #   f"{instruction}\n### Code：\n{example['output']}<|endoftext|>",
#             prompt,
#             padding="max_length",
#             truncation=True,
#             max_length=512,
#             return_tensors="pt"
#         )
        
        
#         return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0)}
#     return dataset.map(format_example, remove_columns=dataset.column_names)

def process_data(tokenizer):
    raw_dataset = load_dataset("json", data_files=data_path)["train"]
    # 快速验证，仅取一小部分数据
    # raw_dataset = raw_dataset.select(range(40))  # 总共只用40条

    
    # 90% 训练，10% 验证
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    def format_example(example):
        code = example['output']
        match = re.search(r'^(def\s+\w+\s*\(.*?\):)', code, re.MULTILINE)
        if not match:
            match = re.search(r'^(class\s+\w+\s*(\(.*?\))?:)', code, re.MULTILINE)
        if match:
            definition_line = match.group(1)
        else:
            lines = [line.strip() for line in code.splitlines() if line.strip()]
            definition_line = lines[0] if lines else "# No code found"
        prompt = f"{definition_line}  # {example['instruction']}"
        inputs = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0)
        }

    train_dataset = train_data.map(format_example, remove_columns=train_data.column_names)
    val_dataset = val_data.map(format_example, remove_columns=val_data.column_names)
    return train_dataset, val_dataset


# LoRA配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=2,  # 显存优化设置 
    gradient_accumulation_steps=4,  # 累计梯度相当于batch_size=8
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,  # 开启混合精度
    logging_steps=20,
    save_strategy="no",
    eval_strategy="epoch",
    eval_steps=2,
    report_to="none",
    optim="adamw_torch",
    no_cuda=False,  # 强制使用CUDA
   dataloader_pin_memory=False,  # 加速数据加载
   remove_unused_columns=False  # 防止删除未使用的列
)

# def main():    
#     # 创建输出目录 
#     os.makedirs(output_path, exist_ok=True) 
#     # 加载tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path) 
#     tokenizer.pad_token = tokenizer.eos_token    
#     # 加载模型到GPU 
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         device_map={"": device}  # 强制使用指定GPU
#     )
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#     # 准备数据
#     dataset = process_data(tokenizer)
#     # 训练回调
#     loss_callback = LossCallback()
    
#     # 数据加载器
#     def data_collator(data):
#         batch = {
#             "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device),
#             "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]).to(device),
#             "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device)  # 使用input_ids作为labels
#         } 
#         return batch 
#     # 创建Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         data_collator=data_collator,
#         callbacks=[loss_callback]
#     )
#     # 开始训练
#     print("开始训练...")
#     trainer.train()
#     # 保存最终模型
#     trainer.model.save_pretrained(output_path)
#     print(f"模型已保存至：{output_path}")
#     # 绘制训练集损失Loss曲线
#     plt.figure(figsize=(10, 6))
#     plt.plot(loss_callback.losses)
#     plt.title("Training Loss Curve")
#     plt.xlabel("Steps")
#     plt.ylabel("Loss")
#     plt.savefig(os.path.join(output_path, "loss_curve.png"))
#     print("Loss曲线已保存")
    
def main():
    os.makedirs(output_path, exist_ok=True) 
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    tokenizer.pad_token = tokenizer.eos_token    

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # datasets = process_data(tokenizer)
    # train_dataset = datasets["train"]
    # eval_dataset = datasets["eval"]
    train_dataset, eval_dataset = process_data(tokenizer)
    loss_callback = LossCallback()

    def data_collator(data):
        return {
            "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device),
            "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]).to(device),
            "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device)
        }

    # 更新 training_args 添加 evaluation_strategy
    # training_args.evaluation_strategy = "steps"
    # training_args.eval_steps = 2  # 可自定义
    # training_args.save_strategy = "no"  # 如果你不想保存中间checkpoint

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # callbacks=[loss_callback]
        callbacks=[loss_callback, EvalLossCallback()]

    )

    print("开始训练...")
    trainer.train()

    trainer.model.save_pretrained(output_path)
    print(f"模型已保存至：{output_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    print("Loss曲线已保存")

if __name__ == "__main__":
     main() 