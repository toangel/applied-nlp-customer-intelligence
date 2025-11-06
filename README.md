# Email Classification with RoBERTa – From Insights to Agentic AI 🚀

This project demonstrates how a **RoBERTa large language model** can be fine-tuned to classify large volumes of inbound email communications and uncover actionable operational insights.

### 🌍 Business Context
At Viator (a Tripadvisor company), millions of supplier and customer emails are received globally each year.  
Manually categorizing these messages was time-consuming and inconsistent, limiting our ability to identify root causes of contact.

To solve this, we built and deployed an **automated email classification system** powered by **RoBERTa-large**, capable of analyzing communications and identifying the main drivers of operator contact — improving customer experience and operational efficiency.

The insights from this model directly inspired **Viator’s first Agentic AI project**, where an AI agent automatically processes simple supplier-initiated refund requests end-to-end.

---

### 🧩 Project Overview

| Component | Description |
|------------|-------------|
| **Model** | RoBERTa-large fine-tuned for multi-class text classification |
| **Input** | Cleaned and anonymized email text |
| **Output** | Predicted theme/category (e.g., refund request, change booking, pricing issue) |
| **Libraries** | HuggingFace Transformers, PyTorch, Scikit-learn, Pandas |
| **Deployment** | Model hosted via internal API endpoint for daily classification |
| **Impact** | Enabled prioritization of operator experience improvements and AI automation use cases |

---

### 🧰 Setup

```bash
git clone https://github.com/<your-username>/email-classification-agentic-ai.git
cd email-classification-agentic-ai
pip install -r requirements.txt
---

### 🧪 Training Example

Here’s an example of how the RoBERTa model is fine-tuned on anonymized email data.  
You can find the complete code inside the [RobertaTraining.ipynb](notebooks/RobertaTraining.ipynb) notebook.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# Load data
df = pd.read_csv("data/sample_emails.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(df["email_text"], df["theme"], test_size=0.2)

# Tokenization
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# Label encoding
label2id = {label: i for i, label in enumerate(df["theme"].unique())}
train_labels = [label2id[label] for label in train_labels]
val_labels = [label2id[label] for label in val_labels]

# Dataset class
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# Prepare datasets
train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

# Initialize model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label2id))

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
