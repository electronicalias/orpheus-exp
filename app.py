from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import torch
from snac import SNAC

model_name = "unsloth/orpheus-3b-0.1-pretrained"
model_path = "models/orpheus"
print(f"Using Model: {model_name}")

print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Loaded Tokenizer from File")

print("Setting device to cuda")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device now set to {device}")

print("Loading snac")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
print("Completed loading snac")

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')
print("Completed Loading Model")