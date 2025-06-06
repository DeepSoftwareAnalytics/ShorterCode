# ShortCoder

We propose a novel approach to generate shorter code using LLMs.
![1](approach.pdf)

Our contributions can be summarized as follows:
* We present and publicly release ShorterCodeBench, a high-quality code brevity optimization dataset comprising 828 carefully curated <original code, simplified code> pairs.
* We proposed ShortCoder, which can solve problems while generating as short code as possible, achieving a 18.1% improvement in the generation efficiency of LLMs.
* We perform an extensive evaluation of ShortCoder. Experimental results show that ShortCoder outperforms the state-of-the-art methods.

## Source code 
### Environment
```
conda create -n ShorterCode python=3.10.13
conda activate ShorterCode

# CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# transformers 4.34.1
pip install transformers==4.34.1

# peft 0.6.2
pip install peft==0.6.2

# bitsandbytes 0.42.0
pip install bitsandbytes==0.42.0

pip install accelerate==0.24.1 appdirs==1.4.4 loralib==0.1.2 black==23.11.0 datasets==2.14.6 fire==0.5.0 sentencepiece==0.1.99 jsonlines==4.0.0
```

### Data

```
cd dataset
```

#### Fine-tuning


```
python finetune.py \
--base_model 'codellama/CodeLlama-7b-Instruct-hf' \
--data_path '../dataset/train.jsonl' \
--output_dir '../output' \
--batch_size 256 \
--micro_batch_size 16 \
--num_epochs 2 \
--val_set_size 0.1

```

#### Reference

```
python inference.py \
--load_8bit \
--base_model 'codellama/CodeLlama-7b-Instruct-hf' \
--lora_weights '../output'
 
```
