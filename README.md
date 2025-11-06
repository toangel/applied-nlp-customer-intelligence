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
