# Applied NLP for Customer Intelligence

An applied NLP case study by Angel Tang. I built a pipeline that analysed **1.48 million customer interactions in one year**, classified them into **15 contact-reason themes**, and uncovered **108 recurring issues** that Product and Operations teams could investigate.

**[Read the full technical case study](https://toangel.github.io/applied-nlp-customer-intelligence/index.html)**

## Business problem

Manually entered Salesforce contact reasons were inconsistent, making it difficult to understand why customers contacted support and which problems were worth fixing first. The pipeline turned unstructured conversations into a consistent view of contact reasons for trend reporting, investigation and prioritisation.

## How it works

| Step | Approach |
| --- | --- |
| Prepare conversations | Clean email, live chat and call transcripts by channel, retaining early customer-authored content. Use BART-large-CNN or FLAN-T5-base to extract the primary reason for contact. |
| Build the taxonomy | Use GPT-4o to propose labels on sampled intents and Claude Sonnet to label the same sample independently. Review disagreements with humans and refine the taxonomy into 15 themes. |
| Classify at scale | Fine-tune RoBERTa-large to assign each extracted intent to one theme. This keeps reporting consistent and limits large-LLM inference to the taxonomy design stage. |
| Discover recurring issues | Use sentence embeddings and cosine similarity to group related intents within each theme. For example, *Update itinerary* breaks down into duration mismatches, start-time changes and requests to add a departure. |

The production data pipeline used **Python, SQL and BigQuery**.

## Evaluation and monitoring

The evaluation approach combined a stratified, human-labelled sample with **precision and recall by theme**, review of commonly confused labels, and checks on reviewer agreement. For granular issue grouping, reviewers sampled examples to assess whether they described the same underlying problem and pointed to a specific action. Production monitoring tracked shifts in theme volumes and prediction confidence.

## Business impact

Product and Operations could track contact reasons over time, investigate spikes and prioritise product fixes, self-service and automation opportunities. One finding helped identify recurring refund requests suitable for automation. The resulting refund-automation initiative supported **more than $350,000 in annual savings**.

## My contribution

I led the design and delivered most of the technical implementation, from problem framing and data preparation through taxonomy design, modelling and validation. I worked with Data Science, Product and Operations colleagues to turn the findings into prioritised opportunities.
