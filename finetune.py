from huggingface_hub import notebook_login

notebook_login()

"""
!pip install -q -U bitsandbytes
!pip install transformers==4.31
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q datasets
"""

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline,TextStreamer

from peft import prepare_model_for_kbit_training,LoraConfig, get_peft_model,PeftModel

from datasets import load_dataset

def getconfig():
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )

  lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    #target_modules=["query_key_value"],
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"], #specific to Llama models.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
  )
  return bnb_config, lora_config

def getmodeltoken(model_id):
    bnb_config, _= getconfig()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    
    return (model,tokenizer)
    
def getnames(model_id,author="TimelyFormulation74"):
    base_model_name = model_id.split("/")[-1]
    adapter_model = f"{author}/{base_model_name}-fine-tuned-adapters"
    new_model = f"{author}/{base_model_name}-fine-tuned"
    return (adapter_model,new_model)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def mystream(user_prompt,model,tokenizer):
    model.config.use_cache = True
    model.eval()
    runtimeFlag = "cuda:0"
    system_prompt = 'You are a helpful assistant that provides accurate and concise responses'
    streamer = TextStreamer(tokenizer)
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)
    
    res = model.generate(**inputs, streamer=streamer, max_new_tokens=500)
    print(tokenizer.decode(res[0]))
    return res


def checkmodel(model_id,prompt):
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    #bnb_config, _= getconfig()
    #model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    
    model,tokenizer=getmodeltoken(model_id)

    #generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    #generator("What is Aristotle's approach to logic?")

    mystream(prompt,model,tokenizer)

    del model

def savetohub(model_id,model,author="TimelyFormulation74"):
    base_model_name = model_id.split("/")[-1]
    adapter_model , new_model = getnames(model_id,author)
    
    #model.save_pretrained(adapter_model, push_to_hub=True, use_auth_token=True)
    #model.push_to_hub(adapter_model, use_auth_token=True)
    
    #model.push_to_hub(new_model, use_auth_token=True, max_shard_size="5GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.push_to_hub(new_model, use_auth_token=True)

    return new_model


def myTrain(model_id,data):
    bnb_config, lora_config= getconfig()
    
    model,tokenizer=getmodeltoken(model_id)

    data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    tokenizer.pad_token = tokenizer.eos_token # </s>
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=20,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False 
    trainer.train()
    adapter_model, newmodel_id = getnames(model_id)
    model = PeftModel.from_pretrained(
        model,
        adapter_model,
    )
    model = model.merge_and_unload()
    
    #newmodel_id=savetohub(model_id,model)

    return (model,tokenizer,newmodel_id)



model_id = "meta-llama/Llama-2-7b-chat-hf" 

data = load_dataset("TimelyFormulation74/test")


model,tokenizer,_=myTrain(model_id,data)
