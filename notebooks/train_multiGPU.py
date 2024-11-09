# ! pip install transformers==4.38.1
# ! pip install rdkit==2023.9.4
# ! pip install accelerate==0.27.2
# ! pip install flash-attn
# ! pip install -q -U bitsandbytes
# ! pip install datasets
# ! pip install loralib
# ! pip install git+https://github.com/huggingface/peft.git
# ! pip install sentencepiece

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install('torch==2.1.0')
install('transformers==4.38.1')
install('rdkit==2023.9.4')
install('accelerate==0.27.2')
install('flash-attn')
install('bitsandbytes')
install('datasets')
install('loralib')
install('git+https://github.com/huggingface/peft.git')
install('sentencepiece')


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
import random, pickle, json, os
from datasets import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb
from peft import PeftModelForCausalLM, get_peft_model, LoraConfig
from torch.cuda.amp import autocast

import sys
sys.path.append('../credentials/')
from HF_credentials import *

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, BitsAndBytesConfig

llm_tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', token=HF_CREDENTIALS, model_max_length=256, add_prefix_space=False)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = "right"

chat = [
  {"role": "user", "content": ""},
  {"role": "assistant", "content": ""}
]

llm_tokenizer.apply_chat_template(chat, tokenize=False)

def create_datasets(split='train'):

    conversations = []
    input_smiles = []

    with open(f'./data/LlaSMol/{split}/property_prediction-bbbp.jsonl', 'r') as f:
        for line in f:
            txt = json.loads(line)
            chat[0]['content'] = f"Is blood-brain barrier permeability (BBBP) a property of <SMILES> {txt['input']} </SMILES>?"
            chat[1]['content'] = f"<BOOLEAN> {txt['output']} </BOOLEAN>"
            # conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=True, truncation=True, padding='max_length', max_length=256))
            conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=False))
            input_smiles.append(txt['input'])
    print(conversations[-1])

    with open(f'./data/LlaSMol/{split}/property_prediction-clintox.jsonl', 'r') as f:
        for line in f:
            txt = json.loads(line)
            chat[0]['content'] = f"Is <SMILES> {txt['input']} </SMILES> toxic?"
            chat[1]['content'] = f"<BOOLEAN> {txt['output']} </BOOLEAN>"
            # conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=True, truncation=True, padding='max_length', max_length=256))
            conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=False))
            input_smiles.append(txt['input'])
    print(conversations[-1])

    with open(f'./data/LlaSMol/{split}/property_prediction-esol.jsonl', 'r') as f:
        for line in f:
            txt = json.loads(line)
            chat[0]['content'] = f"How soluble is <SMILES> {txt['input']} </SMILES>?"
            chat[1]['content'] = f"Its log solubility is <NUMBER> {txt['output']} </NUMBER> mol/L"
            # conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=True, truncation=True, padding='max_length', max_length=256))
            conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=False))
            input_smiles.append(txt['input'])
    print(conversations[-1])

    with open(f'./data/LlaSMol/{split}/property_prediction-hiv.jsonl', 'r') as f:
        for line in f:
            txt = json.loads(line)
            chat[0]['content'] = f"Can <SMILES> {txt['input']} </SMILES> serve as an inhibitor of HIV replication?"
            chat[1]['content'] = f"<BOOLEAN> {txt['output']} </BOOLEAN>"
            # conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=True, truncation=True, padding='max_length', max_length=256))
            conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=False))
            input_smiles.append(txt['input'])
    print(conversations[-1])

    with open(f'./data/LlaSMol/{split}/property_prediction-lipo.jsonl', 'r') as f:
        for line in f:
            txt = json.loads(line)
            chat[0]['content'] = f"Predict the octanol/water distribution coefficient logD under the circumstances of pH 7.4 for <SMILES> {txt['input']} </SMILES>"
            chat[1]['content'] = f"<NUMBER> {txt['output']} </NUMBER>"
            # conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=True, truncation=True, padding='max_length', max_length=256))
            conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=False))
            input_smiles.append(txt['input'])
    print(conversations[-1])

    with open(f'./data/LlaSMol/{split}/property_prediction-sider.jsonl', 'r') as f:
        for line in f:
            txt = json.loads(line)
            chat[0]['content'] = f"Are there any known side effects of <SMILES> {txt['input']} </SMILES> affecting the heart?"
            chat[1]['content'] = f"<BOOLEAN> {txt['output']['Vascular disorders']} </BOOLEAN>"
            # conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=True, truncation=True, padding='max_length', max_length=256))
            conversations.append(llm_tokenizer.apply_chat_template(chat, tokenize=False))
            input_smiles.append(txt['input'])
    print(conversations[-1])
    print(len(conversations))

    return conversations, input_smiles

print('Train:')
train_conversations, train_input_smiles = create_datasets('train')
print('Val:')
val_conversations, val_input_smiles = create_datasets('val')
print('Test:')
test_conversations, test_input_smiles = create_datasets('test')


class CombinedDataset(Dataset):
    def __init__(self, smiles_list, conversations, encoder_tokenizer, llm_tokenizer, max_length=256):
        self.smiles_list = smiles_list
        self.conversations = conversations
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        smiles_encoding = self.encoder_tokenizer(smiles, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        conversation_tokenized = self.llm_tokenizer(self.conversations[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt', add_special_tokens=False)
        return {key: tensor[0].to('cuda') for key, tensor in smiles_encoding.items()}, {key: tensor[0].to('cuda') for key, tensor in conversation_tokenized.items()}#conversation_tokenized.to('cuda')
    
# Load tokenizers
chemberta_tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
mistral_tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', add_prefix_space=False)
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_tokenizer.padding_side = "right"

# Create combined dataset
train_dataset = CombinedDataset(train_input_smiles, train_conversations, chemberta_tokenizer, mistral_tokenizer)
val_dataset = CombinedDataset(val_input_smiles, val_conversations, chemberta_tokenizer, mistral_tokenizer)
test_dataset = CombinedDataset(test_input_smiles, test_conversations, chemberta_tokenizer, mistral_tokenizer)

# Define DataLoader
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class MolEncoderLLMPipeline(nn.Module):
    def __init__(self, lora_rank=32, lora_alpha=64):
        super().__init__()
        # Load molecule encoder
        self.mol_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR", torch_dtype=torch.bfloat16)#.to('cuda:0')

        # UNCOMMENT TO BRING DOWN FROM 15GB TO 7GB
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= False,
        )
        self.llm_config = AutoConfig.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', token=HF_CREDENTIALS)
        self.llm_model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2',
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            token=HF_CREDENTIALS
        )#.to('cuda:1')

        self.llm_model.config.use_cache = False
        self.llm_model.config.pretraining_tp = 1

        # Initialize LoRA layers for Mistral
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Freeze encoder and LLM weights
        for param in self.mol_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.linear_project = nn.Linear(self.mol_encoder.config.hidden_size, self.llm_config.hidden_size, dtype=torch.bfloat16)#.to('cuda:0')

        # Apply LoRA modification
        self.llm_model = get_peft_model(self.llm_model, self.lora_config)
        # self.llm_model.to('cuda:1')

    def forward(self, smiles_tokens, text_tokens):
        # Encoder forward pass / Get SMILES embeddings
        # smiles_tokens = {k: v.to('cuda') for k, v in smiles_tokens.items()}
        # text_tokens = {k: v.to('cuda') for k, v in text_tokens.items()}

        mol_encoder_output = self.mol_encoder(**smiles_tokens)
        smiles_embedding = mol_encoder_output['last_hidden_state'][:,0,:] # torch.Size([batch, max_length, 384])
        smiles_projection = self.linear_project(smiles_embedding).unsqueeze(1)#.to('cuda:1')
        # print('smiles proj')
        # print(smiles_projection.shape)

        # Get embeddings from LLM for the question
        embedding_layer = self.llm_model.model.model.embed_tokens
        llm_embeddings = embedding_layer(text_tokens['input_ids']).squeeze(1)#.to('cuda:1') # torch.Size([batch, max_length, 4096])
        # print('llm emb')
        # print(llm_embeddings.shape)

        # Concatenate encoder and LLM embeddings
        combined_embeddings = torch.cat((smiles_projection, llm_embeddings), dim=1)#.to('cuda:1')
        # print(combined_embeddings.shape)

        # Custom attention mask
        attention_mask = torch.zeros(smiles_projection.shape[0], combined_embeddings.shape[1], combined_embeddings.shape[1], device='cuda')
        attention_mask[:, 0, 0] = 1 # SMILES mask for itself
        for i in range(1, combined_embeddings.shape[1]):
            attention_mask[:, i, 0:i+1] = 1 # From SMILES to current token (inclusive)

        attention_mask = attention_mask.unsqueeze(1)
        # print(attention_mask.shape)

        # Pass through Mistral's transformer layers with LoRA adjustments
        output = self.llm_model(inputs_embeds=combined_embeddings, attention_mask=attention_mask)

        return output
    
    
# # Assuming that we are on a machine with 2 GPUs.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = MolEncoderLLMPipeline(lora_rank=16, lora_alpha=16)
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

# model = MolEncoderLLMPipeline(lora_rank=16, lora_alpha=16).to('cuda')


import gc
gc.collect()
torch.cuda.empty_cache()

def parse_answer(text, is_boolean):
    if is_boolean:
        start = text.find('<BOOLEAN>') + len('<BOOLEAN>')
        end = text.find('</BOOLEAN>')
        return text[start:end].strip()
    else:
        start = text.find('<NUMBER>') + len('<NUMBER>')
        end = text.find('</NUMBER>')
        return text[start:end].strip()

def get_answer(true_sentence, pred_sentence):
    try:
        true_answer = true_sentence.split('[/INST]')[1]
        pred_answer = pred_sentence.split('[/INST]')[1]

        if 'BOOLEAN' in true_answer:
            y_true = parse_answer(true_answer, True)
            y_pred = parse_answer(pred_answer, True)
            return y_true, y_pred

        elif 'NUMBER' in true_answer:
            y_true = parse_answer(true_answer, False)
            y_pred = parse_answer(pred_answer, False)
            return y_true, y_pred
    except:
        return 'Wrong', 'Wrong'
    
def main(rank, world_size):
    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # Define your model
    model = MolEncoderLLMPipeline(lora_rank=16, lora_alpha=16)
    model = model.to(device_ids[0])
    model = DistributedDataParallel(model, device_ids=device_ids)

    
    
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=mistral_tokenizer.pad_token_id)

    # Define the total number of training steps and the number of warmup steps
    epochs = 10
    total_steps = len(test_loader) * epochs
    warmup_steps = 1000

    accumulation_steps = 4

    # Create the learning rate scheduler
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(epochs):
        total_loss = 0
        tprog = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in tprog:
            model.train();
            smiles_data, conversation_data = batch

            # Forward pass
            with autocast():
                output = model(smiles_data, conversation_data)
                logits = output.logits[:, 1:, :]

                # Prepare labels
                labels = conversation_data['input_ids'].squeeze(1)
                labels = torch.cat([labels[:, 1:], labels.new_full((labels.size(0), 1), mistral_tokenizer.pad_token_id)], dim=1)

                # Compute loss
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.view(-1))

            # Backward and accumulate gradients
            loss.backward()
            total_loss += loss.item()
            tprog.set_description(f'train step loss: {loss.item():.4f}')

            if (i+1) % accumulation_steps == 0:  # Step the optimizer every accumulation_steps
                optimizer.step()
                optimizer.zero_grad()

                # Step the scheduler
                scheduler.step()

                # Clean
                gc.collect()
                torch.cuda.empty_cache()

            # Validation step
            if (i % 5000 == 0):# & (i != 0):
                with torch.no_grad():
                    model.eval();

                    categories = ["BBBP", "side effects", "logD", "soluble", "toxic", "HIV"]
                    preds, invalid_count, trues = {cat: [] for cat in categories}, {cat: 0 for cat in categories}, {cat: [] for cat in categories}

                    def convert_to_boolean(input_string):
                        return True if input_string == 'Yes' else False if input_string == 'No' else None

                    val_dataset = CombinedDataset(val_input_smiles[:100], val_conversations[:100], chemberta_tokenizer, mistral_tokenizer)
                    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

                    for batch, true_sentence in zip(val_loader, val_conversations):
                        # Predict
                        smiles_data, conversation_data = batch
                        output = model(smiles_data, conversation_data)
                        output_ids = output.logits.argmax(dim=-1)
                        pred_sentence = mistral_tokenizer.decode(output_ids.tolist()[0])
                        y_true, y_pred = get_answer(true_sentence, pred_sentence)
                        for category in categories:
                            if category in true_sentence:
                                if category in ["BBBP", "side effects", "toxic", "HIV"]:  # binary categories
                                    if y_pred in ['Yes', 'No']:
                                        preds[category].append(convert_to_boolean(y_pred))
                                        trues[category].append(convert_to_boolean(y_true))
                                    else:
                                        invalid_count[category] += 1
                                else:  # continuous categories
                                    try:
                                        preds[category].append(float(y_pred))
                                        trues[category].append(float(y_true))
                                    except:
                                        invalid_count[category] += 1

                    for key in preds:
                        if len(preds[key]) > 0:  # to avoid division by zero
                            if key in ["BBBP", "side effects", "toxic", "HIV"]:  # binary categories
                                accuracy = accuracy_score(trues[key], preds[key])
                                print(f'{key} accuracy: {accuracy:.4f}')
                            else:  # continuous categories
                                rmse = sqrt(mean_squared_error(trues[key], preds[key]))
                                print(f'{key} RMSE: {rmse:.4f}')
                    print('Invalid count:')
                    print(invalid_count)

                    # Clean
                    gc.collect()
                    torch.cuda.empty_cache()

        # Save the model
        torch.save(model.state_dict(), f"output/model_{epoch}.pth")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

    
if __name__ == "__main__":

    size = 4  # number of GPUs
    mp.set_start_method('spawn', force=True)  # force set start method to 'spawn'
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    
# if __name__ == "__main__":
#     import torch.multiprocessing as mp
#     size = 4  # number of GPUs
#     with mp.Pool(processes=size) as pool:
#         pool.starmap(init_process, [(rank, size, main) for rank in range(size)])
    
    
# size = 4  # number of GPUs
# # if mp.get_start_method(allow_none=True) != 'spawn':
# mp.set_start_method('spawn', force=True)  # set start method to 'spawn'
# processes = []
# for rank in range(size):
#     p = mp.Process(target=init_process, args=(rank, size, main))
#     p.start()
#     processes.append(p)

# for p in processes:
#     p.join()