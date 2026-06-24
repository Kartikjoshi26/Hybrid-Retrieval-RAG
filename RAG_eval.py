import os
import sys
import time
from dotenv import load_dotenv
from datasets import Dataset 
# RAGAS and LangChain imports
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall, 
    context_entity_recall, 
    answer_similarity, 
    answer_correctness
)
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# Project imports
from vector_store_utils import get_vector_store
from retriever_utils import hybrid_retriever_with_compression
from security_utils import anonymize_text, deanonymize_text, process_and_sanitize_docs, check_prompt_safety_api
from prompt_utils import set_System_prompt
# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# Fallback to llama-3.3-70b-versatile if RAG_EVAL_MODEL_ID is not set in .env
RAG_EVAL_MODEL_ID = os.getenv("RAG_EVAL_MODEL_ID", "openai/gpt-oss-120b")
RAG_CHATBOT_MODEL_ID = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
print("Initializing RAG components...")
# 2. Initialize LLM and Embedding Model
import threading

class PatchedChatGroq(ChatGroq):
    def __init__(self, *args, **kwargs):
        # Load keys from environment
        api_keys = [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3")
        ]
        # Filter out None or empty keys
        api_keys = [k for k in api_keys if k]
        
        if not api_keys and "groq_api_key" in kwargs:
            api_keys = [kwargs["groq_api_key"]]
            
        if api_keys:
            kwargs["groq_api_key"] = api_keys[0]
        else:
            kwargs["groq_api_key"] = "gsk_dummy_key"
            api_keys = ["gsk_dummy_key"]
            
        super().__init__(*args, **kwargs)
        
        # Use object.__setattr__ to bypass Pydantic field validation for private attributes
        object.__setattr__(self, "_api_keys", api_keys)
        
        sub_llms = []
        for key in api_keys:
            sub_kwargs = kwargs.copy()
            sub_kwargs["groq_api_key"] = key
            sub_llms.append(ChatGroq(*args, **sub_kwargs))
            
        object.__setattr__(self, "_sub_llms", sub_llms)
        object.__setattr__(self, "_current_index", 0)
        object.__setattr__(self, "_lock", threading.Lock())
        print(f"PatchedChatGroq successfully initialized with {len(sub_llms)} rotating keys.")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if "n" in kwargs:
            kwargs["n"] = 1
            
        if not self._sub_llms:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            
        attempts = len(self._sub_llms)
        last_exception = None
        for _ in range(attempts):
            with self._lock:
                current_idx = self._current_index
                llm = self._sub_llms[current_idx]
                
            try:
                res = llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                with self._lock:
                    object.__setattr__(self, "_current_index", (current_idx + 1) % len(self._sub_llms))
                return res
            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "429" in err_msg or "rate limit" in err_msg:
                    print(f"[Rotation Sync] Key index {current_idx} rate limited. Trying next key...")
                    with self._lock:
                        object.__setattr__(self, "_current_index", (current_idx + 1) % len(self._sub_llms))
                    continue
                else:
                    raise e
        raise last_exception

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if "n" in kwargs:
            kwargs["n"] = 1
            
        if not self._sub_llms:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            
        attempts = len(self._sub_llms)
        last_exception = None
        for _ in range(attempts):
            with self._lock:
                current_idx = self._current_index
                llm = self._sub_llms[current_idx]
            
            try:
                res = await llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                with self._lock:
                    object.__setattr__(self, "_current_index", (current_idx + 1) % len(self._sub_llms))
                return res
            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "429" in err_msg or "rate limit" in err_msg:
                    print(f"[Rotation Async] Key index {current_idx} rate limited. Trying next key...")
                    with self._lock:
                        object.__setattr__(self, "_current_index", (current_idx + 1) % len(self._sub_llms))
                    continue
                else:
                    raise e
        raise last_exception

# The judge LLM for RAGAS evaluation
llm = PatchedChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name=RAG_EVAL_MODEL_ID
)

# The actual RAG chatbot LLM for generating responses
rag_llm = PatchedChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name=RAG_CHATBOT_MODEL_ID
)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# 3. Load Evaluation Data (URLs)
urls = [
    "https://en.wikipedia.org/wiki/New_York_City",
    "https://en.wikipedia.org/wiki/Snow_leopard",
    "https://www.britannica.com/place/Galapagos-Islands",
    "https://www.birdlife.org/birds/penguins/#:~:text=The%20threats%20are%20numerous%2C%20including,is%20melting%20before%20their%20eyes."
]


loader = SeleniumURLLoader(urls=urls)
documents = loader.load()


vector_store = get_vector_store(embedding_model)
retriever = hybrid_retriever_with_compression(vector_store, documents, k=5)


queries = [
    "Who discovered the Galapagos Islands and how?",
    "What is Brooklyn–Battery Tunnel?",
    "Are Penguins found in the Galapagos Islands?",
    "How many languages are spoken in New York?",
    "In which countries are snow leopards found?",
    "What are the threats to penguin populations?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "How did Galapagos Islands get its name?",
    "What is the significance of the Statue of Liberty in New York City?",
    
    "How many boroughs make up New York City, and what are they?",
    "When was New York City first settled, and who founded it?",
    "When were the five boroughs consolidated into modern New York City?",
    "What is the approximate 2020 population of New York City?",
    "Where is New York City located geographically within the United States?",
    "Why is New York City considered a global center of finance and diplomacy?",
    "Which famous international organization has its headquarters in New York City?",
    "What are some well-known nicknames of New York City?",
    "What natural harbor is New York City located on?",
    "What is the relationship between New York City's boroughs and counties?",

    
    "At what age do snow leopards usually become sexually mature?",
    "During which season do snow leopards usually mate?",
    "How long is the gestation period of a snow leopard?",
    "How many cubs are typically born in a snow leopard litter?",
    "How long do snow leopards usually live in the wild and in captivity?",
    "What kinds of prey do snow leopards hunt in their mountain habitats?",
    "How do snow leopards typically hunt and handle their prey?",
    "What are the major threats to snow leopard populations?",
    "How does livestock grazing contribute to snow leopard decline?",
    "How much snow leopard habitat in the Himalayas may be lost because of climate change according to a cited report?",

    
    "Which country administers the Galapagos Islands?",
    "In which ocean are the Galapagos Islands located, and how far are they from mainland Ecuador?",
    "How many major and smaller islands make up the Galapagos archipelago?",
    "What is the total land area of the Galapagos Islands?",
    "Across how much ocean area are the Galapagos Islands scattered?",
    "What wildlife protection steps did Ecuador take in the Galapagos in 1935 and 1959?",
    "When were the Galapagos Islands designated a UNESCO World Heritage site?",
    "What marine protection measure was created around the Galapagos in 1986?",
    "What is another name for the Galapagos Islands mentioned by Britannica?",
    "Why are the Galapagos Islands geographically notable in relation to the Equator?",

    
    "What family do penguins belong to?",
    "What do penguins mainly eat?",
    "What is the typical lifespan range of penguins according to BirdLife?",
    "What are the group names used for penguins on land and at sea?",
    "Where do most penguins live, and what strategy do they use to stay warm in extreme cold?",
    "How fast can penguins swim when hunting underwater?",
    "What kinds of prey do penguins catch while swimming?",
    "What is unusual about Emperor Penguins breeding compared with other penguin species?",
    "How does the male Emperor Penguin protect the egg during incubation?",
    "In what kinds of habitats can penguins nest besides icy Antarctic areas?"
]


ground_truths = [
    "The Galapagos Islands were discovered in 1535 by the bishop of Panama, Tomás de Berlanga, whose ship had drifted off course while en route to Peru. He named them Las Encantadas (“The Enchanted”), and in his writings he marveled at the thousands of large galápagos (tortoises) found there. Numerous Spanish voyagers stopped at the islands from the 16th century, and the Galapagos also came to be used by pirates and by whale and seal hunters.",
    "The Brooklyn-Battery Tunnel (officially known as the Hugh L. Carey Tunnel) is the longest continuous underwater vehicular tunnel in North America and runs underneath Battery Park, connecting the Financial District in Lower Manhattan to Red Hook in Brooklyn.[586]",
    "Penguins live on the galapagos islands side by side with tropical animals.",
    "As many as 800 languages are spoken in New York.",
    "Siberia, Tajikistan, Kyrgyzstan, Uzbekistan, Kazakhstan, Afghanistan, Pakistan, India, Nepal, Bhutan, Mongolia, and Tibet.",
    "The threats are numerous, including habitat loss, pollution, disease, and reduced food availability due to commercial fishing. Climate change is of particular concern for many species of penguin, as the sea ice that they depend on to find food or build nests is melting before their eyes.",
    "New York City's economic significance is vast, as it serves as the global financial capital, housing Wall Street and major financial institutions. Its diverse economy spans technology, media, healthcare, education, and more, making it resilient to economic fluctuations. NYC is a hub for international business, attracting global companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries, and educational institutions further fuel its economic prowess. The city's transportation network and global influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural epicenter.",
    "New York City got its name when it came under British control in 1664. King Charles II of England granted the lands to his brother, the Duke of York, who named the city New York in his own honor.",
    "Tomás de Berlanga, who discovered the islands, named them Las Encantadas (“The Enchanted”), and in his writings he marveled at the thousands of large galápagos (tortoises) found there. Numerous Spanish voyagers stopped at the islands from the 16th century, and the Galapagos also came to be used by pirates and by whale and seal hunters.",
    "The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.",

    "New York City is made up of five boroughs: Manhattan, Brooklyn, Queens, the Bronx, and Staten Island.",
    "New York City was first settled on May 20, 1624, and it was founded by the Dutch West India Company.",
    "The five boroughs were consolidated into modern New York City on January 1, 1898.",
    "According to the 2020 population figure cited in the source, New York City had about 8,804,190 residents.",
    "New York City is located at the southern tip of New York State on New York Harbor in the northeastern United States.",
    "New York City is considered a global center because it plays a major role in finance, commerce, culture, technology, media, and international diplomacy.",
    "The headquarters of the United Nations is located in New York City.",
    "Some famous nicknames of New York City are 'The Big Apple,' 'The City That Never Sleeps,' and 'Gotham.'",
    "New York City is located on New York Harbor, one of the world's largest natural harbors.",
    "Each of New York City's five boroughs is coextensive with its respective county, meaning every borough corresponds directly to one county.",


    "Snow leopards usually become sexually mature at about two to three years of age.",
    "Snow leopards usually mate in late winter.",
    "The gestation period of a snow leopard is about 90 to 100 days.",
    "A snow leopard litter usually consists of two to three cubs, although in exceptional cases there can be up to seven.",
    "Snow leopards normally live for about 15 to 18 years in the wild and can live up to about 25 years in captivity.",
    "Snow leopards hunt a variety of prey, especially mountain ungulates such as Himalayan blue sheep, and they may also prey on smaller mammals and occasionally livestock.",
    "Snow leopards actively pursue prey down steep mountain slopes, often using the momentum of their initial leap, and then drag the kill to a safer place before feeding.",
    "Major threats to snow leopards include poaching, illegal trade in skins and body parts, loss of natural prey, habitat degradation, climate change, and conflict with people over livestock.",
    "Livestock grazing contributes to snow leopard decline by causing overgrazing, reducing natural prey availability, degrading habitat, and increasing human–wildlife conflict when snow leopards attack domestic animals.",
    "A cited 2012 report noted that climate change could reduce snow leopard habitat in the Himalayas by about 30% by shrinking the alpine zone and shifting the treeline.",

    
    "The Galapagos Islands are administered by Ecuador and form one of its provinces.",
    "The Galapagos Islands lie in the eastern Pacific Ocean, about 600 miles (1,000 km) west of mainland Ecuador.",
    "The Galapagos archipelago consists of 13 major islands and 6 smaller islands, along with many islets and rocks.",
    "The total land area of the Galapagos Islands is about 3,093 square miles (8,010 square km).",
    "The islands are scattered across about 23,000 square miles (59,500 square km) of ocean.",
    "Ecuador designated part of the Galapagos as a wildlife sanctuary in 1935, and in 1959 that sanctuary became Galapagos National Park.",
    "The Galapagos Islands were designated a UNESCO World Heritage site in 1978.",
    "In 1986, the Galapagos Marine Resources Reserve was created to protect the waters surrounding the islands.",
    "Another name for the Galapagos Islands mentioned by Britannica is Archipiélago de Colón; the source also lists names such as Islas de los Galápagos and Las Encantadas.",
    "The Galapagos are geographically notable because they lie on and around the Equator in the eastern Pacific Ocean.",

    
    "Penguins belong to the bird family Spheniscidae.",
    "Penguins are carnivores and mainly eat prey such as krill, squid, and fish.",
    "According to BirdLife, penguins typically live about 15 to 20 years.",
    "A group of penguins may be called a colony, rookery, or waddle on land, and a raft at sea.",
    "Most penguins live around frozen Antarctica, and in extreme cold they huddle together in large groups and rotate positions to conserve heat.",
    "Penguins can swim at speeds of up to about 15 miles per hour when hunting underwater.",
    "While swimming, penguins commonly catch prey such as krill, squid, and fish.",
    "Emperor Penguins are unusual because they are the only penguin species that breed during the cold, dark Antarctic winter.",
    "After the female lays a single egg, the male Emperor Penguin balances it on his feet and keeps it warm under a feathered brood pouch while the female goes on a long hunting trip.",
    "Penguins can nest in many habitats near the ocean, including sea ice, rocky hillsides, temperate rainforests, volcanic islands such as the Galapagos, and beaches in southern Africa."
]



results = []
contexts = []


for query in queries:
    # Retrieve matching documents (no safeguards)
    relevant_docs = retriever.get_relevant_documents(query)
    
    # Build System Prompt using raw query and raw documents
    system_prompt = set_System_prompt(user_input=query, clean_relevant_docs=relevant_docs)
    
    # Generate response
    try:
        response = rag_llm.invoke([system_prompt])
        final_result = response.content
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        final_result = "Error generating response"
        
    results.append(final_result)
    
    # Store the raw context text that was supplied to the LLM
    contents = [doc.page_content for doc in relevant_docs]
    contexts.append(contents)
    
    # Sleep to stay under Groq's 30 RPM (Requests Per Minute) rate limits
    time.sleep(2.0)
# 7. Build RAGAS Dataset
print("Preparing evaluation dataset...")

def strip_sources(text):
    import re
    # Remove inline brackets/parens/brackets with "Source:" inside
    # e.g., 【Source: ...】, [Source: ...], (Source: ...)
    text = re.sub(r'【\s*Source:[^】]*】', '', text)
    text = re.sub(r'\[\s*Source:[^\]]*\]', '', text)
    text = re.sub(r'\(\s*Source:[^\)]*\)', '', text)
    
    # Remove lines or sections starting with Source/Sources/Sources
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip().lower()
        if (stripped.startswith("**source") or 
            stripped.startswith("source:") or 
            stripped.startswith("[source:") or 
            stripped.startswith("sources:") or 
            stripped.startswith("**sources**")):
            break
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()

# Clean generated answers for evaluation
cleaned_results = [strip_sources(r) for r in results]

d = {
    "question": queries,
    "answer": cleaned_results,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(d)
# 8. Evaluate metrics using Groq and the local embedding model
print("Evaluating metrics with RAGAS (throttled to respect rate limits)...")
run_config = RunConfig(
    max_workers=1,      # Process 1 query at a time to stay under 30 RPM limit
    timeout=240,
    max_retries=20,     # If we hit a transient 429 rate limit, wait and retry
    max_wait=60         # Wait up to 60 seconds between retries
)

score = evaluate(
    dataset,
    metrics=[
        faithfulness, 
        answer_relevancy, 
        context_precision, 
        context_recall, 
        context_entity_recall, 
        answer_similarity, 
        answer_correctness
    ],
    llm=llm,
    embeddings=embedding_model,
    run_config=run_config
)


print("Saving evaluation scores...")
score_df = score.to_pandas()
# Keep both the original response with sources and the cleaned response used for evaluation
score_df.insert(3, 'original_response', results)
score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)

print("\nEvaluation Completed!")
print(score_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
                'context_entity_recall', 'answer_similarity', 'answer_correctness']].mean(axis=0))
