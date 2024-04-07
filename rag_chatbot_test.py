
# rag_chatbot_test.py

from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

# Load the RAG model components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)

def get_rag_answer(document, question):
    inputs = tokenizer(question, return_tensors="pt")
    document_inputs = tokenizer(document, return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(document, return_tensors="pt").input_ids

    outputs = model.generate(input_ids=inputs["input_ids"], labels=labels, decoder_input_ids=document_inputs["input_ids"])
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]