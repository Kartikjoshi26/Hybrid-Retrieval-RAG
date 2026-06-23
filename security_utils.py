from presidio_analyzer import PatternRecognizer, Pattern
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_core.documents import Document
from langsmith import traceable

import os
from huggingface_hub import InferenceClient

from concurrent.futures import ThreadPoolExecutor
from typing import List
import streamlit as st
from transformers import pipeline


TARGETED_PII = ["AADHAAR_CARD", "PIN_CODE", "EMAIL_ADDRESS", "CREDIT_CARD"]

anonymizer = PresidioReversibleAnonymizer(analyzed_fields=TARGETED_PII)


aadhaar_pattern = Pattern(name="aadhaar_pattern", regex=r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", score=0.85)
aadhaar_recognizer = PatternRecognizer(supported_entity="AADHAAR_CARD", patterns=[aadhaar_pattern])

pin_pattern = Pattern(name="pin_pattern", regex=r"\b[1-9][0-9]{5}\b", score=0.75)
pin_recognizer = PatternRecognizer(supported_entity="PIN_CODE", patterns=[pin_pattern])

anonymizer.add_recognizer(aadhaar_recognizer)
anonymizer.add_recognizer(pin_recognizer)


@traceable(name = "PII check")
def anonymize_text(text):
    result = anonymizer.anonymize(text)
    return result



@traceable(name = "Undo PII check")
def deanonymize_text(text):
    result = anonymizer.deanonymize(text)
    return result



@traceable(name = 'PII sanitization of the Docs')
def sanitize_docs(docs):
    sanitized_docs = []
    for doc in docs:
        clean_text = anonymizer.anonymize(doc.page_content)
        sanitized_docs.append(Document(page_content = clean_text, metadata=doc.metadata))
    return sanitized_docs





Content_Moderate_MODEL_ID = os.getenv("CONTENT_MODERATE_MODEL_ID", "meta-llama/Llama-Guard-3-8B")
Prompt_Guard_MODEL_ID = os.getenv("PROMPT_GUARD_MODEL_ID", "meta-llama/Llama-Prompt-Guard-2-86M")

@st.cache_resource
def get_prompt_guard_pipeline():
    # Cache Prompt Guard locally so it only loads into memory once
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return pipeline(
        "text-classification",
        model=Prompt_Guard_MODEL_ID,
        token=token
    )



@traceable(name = "Sanitize the text")
def check_prompt_safety_api(user_prompt: str) -> bool:

    client = InferenceClient()

    try:
        ##~~~~~~~~~~~~~~~ with the help local run ~~~~~~~~~~~~~~~~:

        # ## Jailbreak / Prompt Injection detection
        # classifier = get_prompt_guard_pipeline()
        # pg_result = classifier(user_prompt, truncation=True, max_length=512)

        # for res in pg_result:
        #     # LABEL_1 indicates a Malicious / Injection / Jailbreak attempt
        #     # Note: We use dict lookup res['label'] and res['score'] for local inference
        #     if res['label'] == 'LABEL_1' and res['score'] > 0.5:
        #         return False



        ## ~~~~~~~~~~~~~~~ with the help of api which is slow ~~~~~~~~~~~~~~ :

        ## Jailbreak / Prompt Injection detection
        pg_result = client.text_classification(
            user_prompt,
            model=Prompt_Guard_MODEL_ID,
        )
        
        for element in pg_result:
            if element.label == "LABEL_1" and element.score > 0.5:
                return False
        

        ## content moderation check :
        messages = [{"role": "user", "content": user_prompt}]
        
        response = client.chat_completion(
            model=Content_Moderate_MODEL_ID,
            messages=messages,
            max_tokens=10,
            temperature=0.0
        )
        
        verdict = response.choices[0].message.content.strip().lower()
        
        if "unsafe" in verdict:
            return False
        return True
        
    except Exception as e:
        print(f"Hugging Face API Error: {e}")
        return False



@traceable(name="Retrieved Document safety check")
def filter_safe_docs(docs: List[Document]) -> List[Document]:
    """
    Checks retrieved documents for indirect prompt injections in parallel.
    Removes any malicious documents to prevent system hijacking.
    """
    client = InferenceClient()
   
    safe_docs = []

   ##~~~~~~~~~~~~~~~ with the help of local run ~~~~~~~~~~~~~~~~:

    # classifier = get_prompt_guard_pipeline()
    # for doc in docs:
        # try:
        #     # We apply truncation=True to respect the model's 512-token context limit
        #     results = classifier(doc.page_content, truncation=True, max_length=512)
            
        #     # 3. Check labels (results format: [{'label': 'LABEL_1', 'score': 0.99}])
        #     is_safe = True
        #     for res in results:
        #         if res['label'] == 'LABEL_1' and res['score'] > 0.5:
        #             print(f"[Safety Warning] Indirect prompt injection detected in retrieved text (score: {res['score']:.4f})")
        #             is_safe = False
        #             break
            
        #     if is_safe:
        #         safe_docs.append(doc)
                
        # except Exception as e:
        #     print(f"Error checking document safety locally: {e}")
        #     # Fail-safe: if local inference fails, assume the document is safe
        #     safe_docs.append(doc)


        ## ~~~~~~~~~~~~~~~ with the help of api which is slow ~~~~~~~~~~~~~~ :
       
    def check_single_doc(doc: Document) -> tuple[Document, bool]:
        try:
            pg_result = client.text_classification(
                doc.page_content,
                model=Prompt_Guard_MODEL_ID
            )

            for element in pg_result:
                # LABEL_1 is MALICIOUS (jailbreak/injection)
                if element.label == "LABEL_1" and element.score > 0.5:
                    # print(f"[Safety Warning] Indirect prompt injection detected in retrieved text (score: {element.score:.4f})")
                    return doc, False
            return doc, True
        except Exception as e:
            # print(f"Error checking document safety: {e}")
            # Fail-safe: if the API fails, preserve the document
            return doc, True
    # Check all documents in parallel to keep latency minimal
    with ThreadPoolExecutor(max_workers=max(1, len(docs))) as executor:
        results = executor.map(check_single_doc, docs)
    for doc, is_safe in results:
        if is_safe:
            safe_docs.append(doc)
            
    return safe_docs





@traceable(name="Process and Sanitize of retrieved Documents")
def process_and_sanitize_docs(docs: List[Document]) -> List[Document]:

    safe_docs = filter_safe_docs(docs)
    
    clean_docs = sanitize_docs(safe_docs)
    
    return clean_docs   