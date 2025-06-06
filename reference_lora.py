from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
def load_lora_model(base_model_name, lora_path, device="cuda"):
    """
    load base model and lora adapter
    :param base_model_name: name/path of base model
    :param lora_path: path of LoRA adapter
    :param device: running device (cuda/cpu)
    :return: model and tokenizer
    """
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,
        # device_map="auto",
        device_map={"": device}  
    )
    
    # load lora adapter
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()  
    model.save_pretrained("./output/DeepSeek_full")
    tokenizer.save_pretrained("./output/DeepSeek_full")
    model.to(device)
    model.eval()
    
    return model, tokenizer

def build_prompt(instruction, input_context=None):
    """
    construct prompt
    :param instruction: instruction description
    :param input_context: input context
    :return: prompt
    """
    if input_context:
        return f"User: {instruction}\n{input_context}\n\nAssistant: "
    else:
        return f"User: {instruction}\n\nAssistant: "
    

def generate_code(model, tokenizer, prompt, generation_config=None):
    default_config = {
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    if generation_config:
        default_config.update(generation_config)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **default_config
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assistant_start = full_response.find("Assistant: ") + len("Assistant: ")
    code = full_response[assistant_start:].strip()
    
    return code

def main(load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
):
    DEVICE = "cuda"
    BASE_MODEL = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        BASE_MODEL
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    LORA_PATH = lora_weights or os.environ.get("LORA_PATH", "")
    # load model
    model, tokenizer = load_lora_model(BASE_MODEL, LORA_PATH, DEVICE)
    # prompt construction
    instruction = '''
    from typing import List
    def mean_absolute_deviation(numbers: List[float]) -> float:
        """ For a given list of input numbers, calculate Mean Absolute Deviation
        around the mean of this dataset.
        Mean Absolute Deviation is the average absolute difference between each
        element and a centerpoint (mean in this case):
        MAD = average | x - x_mean |
        >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
        1.0
        """
    '''
    
    prompt = build_prompt(instruction)

    generation_config = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }

    generated_code = generate_code(model, tokenizer, prompt, generation_config)
    # show code
    print("="*50)
    print("Code:")
    print("="*50)
    print(generated_code)
   
    
if __name__ == "__main__":
     main() 