from presidio_analyzer import PatternRecognizer, Pattern
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_core.documents import Document
from langsmith import traceable

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

