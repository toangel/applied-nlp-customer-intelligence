# Applied NLP for Customer Intelligence

A two-stage NLP pipeline that analysed **1.48 million customer interactions** at Viator, producing **15 contact-reason themes** for trend reporting and surfacing **108 granular issues** for root-cause analysis. The pipeline used Python, SQL and BigQuery.

```text
Email, live chat and call transcripts
  → Channel-specific cleaning and intent extraction
  → Stage 1 · RoBERTa-large theme classification
  → Stage 2 · Within-theme grouping by cosine similarity
```

**Shared preprocessing.** Keep the first customer email, the first 10 customer chat messages or the early customer speech from a call. Strip channel-specific noise such as quoted replies, system messages and agent dialogue. Use BART-large-CNN for summarisation or FLAN-T5-base for instruction-based extraction to produce a standardised statement of the customer’s primary intent.

## Stage 1 · Classify the contact reason

Build the taxonomy on a sample of extracted intents. GPT-4o proposes issue labels and themes, Claude Sonnet independently labels the same intents, and human reviewers resolve disagreements and consolidate the taxonomy into 15 themes.

Fine-tune **RoBERTa-large** to assign each extracted intent to one theme. This confines large-LLM inference to taxonomy discovery and gives high-volume classification a stable set of labels for reporting.

## Stage 2 · Discover granular issues within each theme

Encode each intent as a **sentence embedding**. Rank neighbours within its predicted theme by **cosine similarity** and group semantically similar intents into recurring issues. For example, *Update itinerary* splits into duration mismatches, start-time changes and requests for additional departures.

See the [technical case study](index.html) for architecture, examples and evaluation criteria.
