#Catalan to english
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torchviz import make_dot
import torch


ds = load_dataset("Helsinki-NLP/opus_books", "ca-en")


model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(translation):
    inputs = ["Translate to Catalan " + row["en"] for row in translation["translation"]]
    targets = [row["ca"] for row in translation["translation"]]
    return tokenizer(inputs, text_target=targets, truncation=True, padding="max_length",max_length=128)

tokenized = ds["train"].map(preprocess, batched=True)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="none",
     dataloader_pin_memory=False

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

trainer.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

inputs = tokenizer("Translate to Catalan Hello, how are you?", return_tensors="pt").to(device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=50)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated Translation:", output_text)

inputs = tokenizer("Translate to Catalan Hello, how are you?", return_tensors="pt").to("cpu")
decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], device="cpu")  # simulate one token decoder input

outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], decoder_input_ids=decoder_input_ids)

dot = make_dot(outputs.logits, params=dict(model.named_parameters()))
dot.render("Catalan translation_graph", format="png")