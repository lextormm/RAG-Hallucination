# Hallucination Detection and Mitigation in AI Systems

## Understanding Hallucinations

AI hallucinations occur when a language model generates content that is factually incorrect, inconsistent, or unsupported by the provided context. The term "hallucination" is borrowed from psychology, where it refers to perceiving things that don't exist.

### Types of Hallucinations

**Factual Hallucinations**: The model states incorrect facts.
- Example: Claiming Albert Einstein won the Nobel Prize in 1945 (he won it in 1921)
- Example: Fabricating a research paper citation that doesn't exist

**Contextual Hallucinations**: The model contradicts the provided context.
- Example: The context says "Company X was founded in 2010" but the model says "founded in 2015"
- Example: Extrapolating beyond what the source documents say

**Intrinsic Hallucinations**: Contradictions within the generated text itself.
- Example: Saying "The speed of light is 300,000 km/s" in one sentence and "the speed of light is 150,000 km/s" in another

**Extrinsic Hallucinations**: Adding information not present in the source.
- Example: Adding extra details beyond what the documents contain

## Hallucination Detection Approaches

### Consistency-Based Detection
The most practical approach: compare the generated answer against the retrieved context documents. This can be done by:
1. Extracting claims from the generated answer
2. Checking each claim against source documents
3. Computing an overall consistency score

### NLI (Natural Language Inference) Based
Using NLI models to determine if the answer is entailed by, contradicts, or is neutral with respect to the context. NLI-based detection classifies relationships as:
- **Entailment**: The context supports the answer
- **Contradiction**: The context contradicts the answer
- **Neutral**: The context doesn't address the answer

### Self-Consistency Checking
Generating multiple responses to the same query and checking if they agree. High variance across responses suggests potential hallucination.

### LLM-as-Judge
Using a separate LLM (or the same LLM in a different prompt) to evaluate whether the generated answer is consistent with the provided context. This is sometimes called "LLM self-reflection."

## Mitigation Strategies

### Prompt Engineering
- Include explicit instructions to "only answer based on the provided context"
- Add negative instructions: "Do not add information not found in the sources"
- Request uncertainty expressions: "If unsure, say so"

### Strict Constraint Regeneration
When hallucination is detected, regenerate with tighter constraints:
- Require explicit source citations for every claim
- Use chain-of-thought to force step-by-step grounding
- Limit response to paraphrasing only what the context says

### Retrieval Quality Improvement
- Improve chunking strategy (smaller, more focused chunks)
- Use better embedding models
- Implement hybrid search (keyword + semantic)
- Add metadata filtering

### Feedback Loops
Implementing iterative correction:
1. Generate initial answer
2. Detect potential hallucinations
3. If detected, regenerate with stricter constraints
4. Repeat up to a maximum number of attempts
5. Return the best answer with confidence score

## Evaluation Metrics

- **Faithfulness Score**: How well the answer is supported by the retrieved context
- **Answer Relevance**: How well the answer addresses the original question
- **Context Recall**: How much of the relevant information was retrieved
- **Hallucination Rate**: Percentage of generated claims not supported by context

Frameworks like RAGAS (RAG Assessment) provide automated evaluation of RAG systems across these dimensions.

## Industry Impact

Hallucination is a critical concern in high-stakes domains:
- **Medical**: False medical information can harm patients
- **Legal**: Fabricated case citations can mislead lawyers
- **Finance**: Incorrect data can lead to bad investment decisions
- **Education**: Students may learn incorrect information

This is why hallucination-aware systems with self-correction mechanisms are essential for production RAG deployments.
