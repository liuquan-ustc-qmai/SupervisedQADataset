
# CoreTech-LLM: A LLM focusing on key technologies and International Patent Classification(IPC) Q&A

## 📖 Introduction
 <!-- CoreTech-LLM是基于Qwen2-7B架构，通过多阶段训练流程构建的领域专用大模型。
本模型采用两阶段优化方案：
​阶段1:​增量预训练阶段​​：在精选的领域语料（涵盖国家专利局IPC及其对应描述文件、专利文本等高质量数据源）上持续训练，增强模型的基础语义理解能力；
​​阶段2:​课程学习微调阶段​​：采用渐进式课程学习（Curriculum Learning）策略，通过难度分级的监督微调，显著提升模型在技术领域的关键技术识别和IPC分类任务中的表现。 -->
**CoreTech-LLM**​ is a domain-specific LLM built upon the Qwen2.5-7B(base) architecture through a multi-stage training pipeline. The model adopts a dual-phase optimization approach:

1. Continue Pre-training (CPT) Phase​​: Continuous training on curated domain corpora (including high-quality data sources such as IPC classification documents from national patent offices with corresponding description files, and patent texts) to enhance the model's fundamental semantic understanding capabilities.

2. Supervised FineTuning (SFT) Phase​​: Implementation of progressive Curriculum Learning (CL) strategy through difficulty-graded supervised fine-tuning, significantly improving the model's performance in technical domain tasks including key technology identification and IPC classification.



<!-- - Hugging Face Demo: doing

We provide a simple Gradio-based interactive web interface. After the service is started, it can be accessed through a browser, enter a question, and the model will return an answer. The command is as follows:
```shell
python gradio_demo.py --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

Parameter Description:

- `--base_model {base_model}`: directory to store LLaMA model weights and configuration files in HF format, or use the HF Model Hub model call name
- `--lora_model {lora_model}`: The directory where the LoRA file is located, and the name of the HF Model Hub model can also be used. If the lora weights have been merged into the pre-trained model, delete the --lora_model parameter
- `--tokenizer_path {tokenizer_path}`: Store the directory corresponding to the tokenizer. If this parameter is not provided, its default value is the same as --lora_model; if the --lora_model parameter is not provided, its default value is the same as --base_model
- `--use_cpu`: use only CPU for inference
- `--gpus {gpu_ids}`: Specifies the number of GPU devices used, the default is 0. If using multiple GPUs, separate them with commas, such as 0,1,2 -->

## 📚 Dataset
### CPT Data
| Classification     | Source           | Nums (Entries) | Tokens    | Storage Size (GB) |
|--------------------|------------------|----------------|------------------|-------------------|
| General Corpus      | wudao            | 260,000        | 235,000,000      | ~0.44 GB          |
| Specialized Corpus  | IPC              | 70,000         | 160,000,000      | ~0.30 GB          |
| Specialized Corpus  | Chinese Patents  | 400,000        | 3,700,000,000    | ~6.90 GB          |
| Specialized Corpus  | European Patents | 50,000         | 912,000,000      | ~1.70 GB          |
| Specialized Corpus  | US Patents       | 50,000         | 1,040,000,000    | ~1.94 GB          |


### SFT Data
| Dataset       | IPC_TASK (Train) | IPC_TASK (Test) | TECH_TASK (Train) | TECH_TASK (Test) |
|---------------|------------------|-----------------|-------------------|------------------|
| KRL-SFT       | 138              | 15              | 138               | 15               |
| GovPatent-SFT | 168              | 22              | 130               | 22               |

## 🚀 Training Pipeline

### Stage 1: Continue Pretraining
<!-- 基于qwen2.5-7b模型，在精选专利语料（含IPC分类文本、专利说明书等）上进行领域适应训练并使用WuDao通用语料进行通用能力预训练得到CoreTech-LLM-base模型。 -->
CoreTech-LLM-base is developed through domain-adaptive training on carefully selected patent corpora (including IPC-classified texts and patent specifications) based on the Qwen2.5-7B model, supplemented with WuDao general corpus for general capability pretraining.


```shell
cd /path/to/LLaMA-Factory

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup bash -c "setsid llamafactory-cli train /path/to/pretrain.yaml" \
> pretrain.log 2>&1 & \
disown -h %%
```


### Stage 2: Supervised FineTuning
<!-- 从高质量文献中以及国家专利文件中提取的领域-关键技术、领域-IPC分类问答对 -->
Based on the CoreTech-LLM-base model, the CoreTech-LLM-instruct model is obtained by using Domain-specific key technologies and domain-IPC classification Q&A pairs extracted from high-quality literature and national patent documents.

```shell
cd /path/to/LLaMA-Factory

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup bash -c "setsid llamafactory-cli train \
/path/to/run_sft.yaml" \
> sft.log 2>&1 & \
disown -h %%
```



## 💾 Install
#### Install the requirements

```markdown
git clone https://github.com/liuquan-ustc-qmai/CoreTech-LLM.git

cd CoreTech-LLM

pip install -r requirements.txt
```

<!-- ### Hardware Requirement (VRAM)


| Train Method  | Bits |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
|-------|------| ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full   | AMP  | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full   | 16   |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| LoRA  | 16   |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA | 8    |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA | 4    |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA | 2    |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB | -->


## 🔥 Inference
After the training is complete, now we load the trained model for local inference.

```shell
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat path/to/run_vllm.yaml

```

<!-- Parameter Description:

- `--base_model {base_model}`: Directory to store LLaMA model weights and configuration files in HF format
- `--lora_model {lora_model}`: The directory where the LoRA file is decompressed, and the name of the HF Model Hub model can also be used. If you have incorporated LoRA weights into the pre-trained model, you can not provide this parameter
- `--tokenizer_path {tokenizer_path}`: Store the directory corresponding to the tokenizer. If this parameter is not provided, its default value is the same as --lora_model; if the --lora_model parameter is not provided, its default value is the same as --base_model
- `--with_prompt`: Whether to merge the input with the prompt template. Be sure to enable this option if loading an Alpaca model!
- `--interactive`: start interactively for multiple single rounds of question and answer
- `--data_file {file_name}`: Start in non-interactive mode, read the contents of file_name line by line for prediction
- `--predictions_file {file_name}`: In non-interactive mode, write the predicted results to file_name in json format
- `--use_cpu`: use only CPU for inference
- `--gpus {gpu_ids}`: Specifies the number of GPU devices used, the default is 0. If using multiple GPUs, separate them with commas, such as 0,1,2 -->


#### Inference Examples
To be added



## ⚠️ LICENSE

To be added

## 😇 Citation
To be added
<!-- If you used MedicalGPT in your research, please cite as follows:

```latex
@misc{MedicalGPT,
   title={MedicalGPT: Training Medical GPT Model},
   author={Ming Xu},
   year={2023},
   howpublished={\url{https://github.com/shibing624/MedicalGPT}},
}
``` -->


<!-- ## 💕 Acknowledgements

- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)
- [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

Thanks for their great work!
#### Related Projects
- [shibing624/ChatPilot](https://github.com/shibing624/ChatPilot): Provide a simple and easy-to-use web UI interface for LLM Agent (including RAG, online search, code interpreter). -->
