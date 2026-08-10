import os
import re
import asyncio
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama




history = []



folder_path = './CSJM_DOCUMENTS/'



all_docs = []
summary = ""
summary_to_docid = {}
docid_to_content = {}
for file in os.listdir(folder_path):
    full_path = os.path.join(folder_path, file)
    if os.path.isfile(full_path):
        doc = Document(page_content= "")
        with open(full_path, "r") as file_content:
            doc.metadata['original_content'] = file_content.read()

        summary_path = os.path.join(folder_path, "Summary", os.path.splitext(file)[0] + "_summary.txt")
        with open(summary_path, "r") as file_content:
            doc.page_content = file_content.read()
            summary += doc.page_content + "\n"
        doc.metadata["source"] = file
        doc.metadata['category'] = os.path.splitext(file)[0]
        
        summary_to_docid[doc.page_content] = doc.metadata['category']
        docid_to_content[doc.metadata['category']] = [doc]
        
        all_docs.append(doc)



embeddings = OllamaEmbeddings(
    model="nomic-embed-text", 
    # base_url="http://ollama-service:11434"
)
vector_store_model = Chroma(
    embedding_function=embeddings,
    persist_directory="../DB_QUEST",
    collection_name="broucher_mode20"
)        
if not vector_store_model.get()['ids']:
    vector_store_model.add_documents(all_docs)




summary_embeddings = {
    did: embeddings.embed_query(summary_key)
    for summary_key, did in summary_to_docid.items()
}    



bm25K = 2
simK = 1
mmrK = 1
# llm = ChatOllama(
#     model="llama3.1:8b",
#     # base_url="http://ollama-service:11434",
#     temperature = 0.9
# )
# llmc = ChatOllama(
#     model="llama3.1:8b",
#     # base_url="http://ollama-service:11434",
#     temperature = 0.7
# )

fir_api = "d88949c209604f50b5ecd9ac0ce43a2c.UDFbsz9fmNuZ4hk2ihaaAQz2"
sec_api = "9e4457cb9a2247a7ac3cfecc62145b43.CdJ6RLJJF4uKMufTmJ4FX2pn"
llm = ChatOllama(
    model="gpt-oss:120b",
    temperature=0.8,
    base_url="https://api.ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {sec_api}"
        }
    }
)
llmc = llm




# ─────────────────────────────────────────
# CHAIN 1 — Query Decomposition
# ─────────────────────────────────────────

decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query decomposition assistant.
Your task:
- Analyze the given user query
- If it is jumbled, complex, or multi-intent, break it into simple, clear, individual queries
- If it is already simple, return it as is
- If it is dependent on a previous query, do not break it down
Rules:
- Do not bias for any course or department if not mentioned
- Each query must be independent and self-contained
- No numbering, no bullet points, no extra text
- Return one query per line and every query should include context from original query
- Maximum 3 queries
- All queries must be in formal English
- Don't separate query without need, if queries are related keep them together
- Don't use punctuation and special characters, convert % sign to percent word"""),
    ("human", """User Query: {user_query}
Decomposed Queries:""")
])

def clean_queries(response: str) -> list[str]:
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    queries = queries[:4]
    if not queries:
        print("[WARNING] Query decomposition failed, using original query")
        queries = [response]
    cleaned = []
    for query in queries:
        query = query.replace("%", "% percent")
        query = re.sub(r'[^a-zA-Z0-9_.% ]', '', query).lower()
        cleaned.append(query)
    return cleaned

decompose_chain = (
    decompose_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_queries)
)

# ─────────────────────────────────────────
# CHAIN 2 — Refine Query (Summary Match)
# ─────────────────────────────────────────

refine_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent university retrieval assistant.
Your tasks:
1. Determine whether the user query matches the provided summary.
2. Classify the query into one of these categories:
- COURSE_ELIGIBILITY
- UNIVERSITY_INFORMATION
- GENERAL_QUERY

Classification Rules:
COURSE_ELIGIBILITY:
- admission based on percentage
- eligible courses
- admission criteria
- qualification requirements

UNIVERSITY_INFORMATION:
- placements, facilities, infrastructure
- hostel, achievements, rankings
- campus life, why join university

GENERAL_QUERY:
- all other queries

Response Rules:
- If the query matches the summary:
    Return the exact summary text as-it is, without any tags or extra formatting.
- If the query does NOT match the summary:
    Return only the word: NO MATCH"""),
    ("human", """Summary:
{summary}

Query:
{query}

Remember: Output ONLY the raw summary text or NO MATCH. No explanation. No formatting.""")
])

refine_chain = refine_prompt | llmc | StrOutputParser()

async def refine_all_queries(queries: list[str]) -> list[str]:
    results = await asyncio.gather(
        *[refine_chain.ainvoke({"query": query, "summary": summary})
          for query in queries]
    )
    return list(results)

# ─────────────────────────────────────────
# CHAIN 3 — Retrievers
# ─────────────────────────────────────────

# Sparse
sparse_retriever = BM25Retriever.from_documents(all_docs)
sparse_retriever.k = bm25K

# Dense
dense_similarity_search = vector_store_model.as_retriever(
    search_type="similarity",
    search_kwargs={"k": simK}
)
dense_mmr_search = vector_store_model.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": mmrK,             
        "fetch_k": 20,      
        "lambda_mult": 0.5  
    }
)

# Ensemble
doc_retriever = EnsembleRetriever(
    retrievers=[sparse_retriever, dense_similarity_search, dense_mmr_search],
    weights=[0.6, 0.2, 0.2]
)

# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a university document router.
Your task is to identify which document best answers the user query.
Available Documents:
- about_csjm: general information about CSJM university, history, vision, mission
- admission_coordinators: contact details of admission coordinators, coordinator names
- admission_process: steps to apply, admission procedure, how to take admission
- all_courses: list of all available courses, programs offered, Engineering MBA BCA Computer Application Legal Studies Pharma e.t.c.
- approved_boards: approved boards for admission, CBSE ICSE UP Board eligibility
- course_eligibility: eligibility criteria for specific courses, qualification requirements, percentage required, fees
- csjm_innovation_foundation: innovation cell, startups, incubation, research foundation
- csjm_placements: placement statistics, campus drives, packages, LPA, recruited companies, Training and Placement Officer
- facilities_at_csjm: infrastructure, labs, library, sports, facilities available on campus
- guidelines_for_admission: admission rules, important guidelines, documents required
- hostel: hostel facilities, accommodation, mess, rooms, hostel fees
- message_from_vice_chancellor: vice chancellor message, VC statement
- teachers_information: faculty details, professor information, teacher contacts
- why_csjm: why choose CSJM, rankings, NAAC, achievements, USP of university
Rules:
- Return ONLY the doc_id from the list above
- No explanation, no extra text, no punctuation
- If query matches multiple documents, return the most relevant one
- If nothing matches, return NO_MATCH"""),
    ("human", """Query: {query}
Doc ID:""")
])

router_chain = router_prompt | llm | StrOutputParser()

VALID_DOC_IDS = {
    "about_csjm", "admission_coordinators", "admission_process",
    "all_courses", "approved_boards", "course_eligibility",
    "csjm_innovation_foundation", "csjm_placements", "facilities_at_csjm",
    "guidelines_for_admission", "hostel", "message_from_vice_chancellor",
    "teachers_information", "why_csjm"
}

async def get_doc_id_by_intent(query: str) -> tuple[str, list] | None:
    result = await router_chain.ainvoke({"query": query})
    result = result.strip().lower()
    
    if result in VALID_DOC_IDS:
        print(f"  Router matched: {result}")
        return query, docid_to_content[result]
    
    print(f"  Router failed: '{result}' — will use fetch_docs result")
    return None

async def fetch_docs(decomposed_query: str, refined_query: str) -> tuple[str, list]:
    if refined_query.strip().upper() == "NO MATCH":
        docs = await doc_retriever.ainvoke(decomposed_query)
        return decomposed_query, docs[:2]
    
    query_embedding = embeddings.embed_query(refined_query)
    scores = []
    for doc_id, summary_emb in summary_embeddings.items():
        score = cosine_similarity(
            [query_embedding],
            [summary_emb]
        )[0][0]
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    best_doc_id, best_score = scores[0]
    
    if best_score >= 0.75:
        print(f"  Summary matched: {best_doc_id} (score: {best_score:.3f})")
        return refined_query, docid_to_content[best_doc_id]
    
    docs = await doc_retriever.ainvoke(decomposed_query)
    return decomposed_query, docs[:2]


# ─────────────────────────────────────────
# CHAIN 4 — Hybrid Prompt + LLM
# ─────────────────────────────────────────

hybrid_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful university assistant.
Answer the query based ONLY on the provided context.
Don't bias for any course if not mentioned.
Use keywords from the query to find relevant information in the context.
If the context does not contain the answer, say 'I dont know the answer'."""),
    ("human", """Context:
---------
{context}
---------

Question: {query}

Answer:""")
])

async def get_answer(decomposed_query: str, context: str) -> str:
    return await (
        hybrid_prompt | llm | StrOutputParser()
    ).ainvoke({
        "context": context,
        "query": decomposed_query      
    })

# ─────────────────────────────────────────
# CHAIN 5 — Final Answer
# ─────────────────────────────────────────


final_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict assistant ONLY for CSJM University (Chhatrapati Shahu Ji Maharaj University), Kanpur.

STRICT RULES:
- ONLY answer questions related to CSJM University.
- but If someone greets you, then you should also greet them and If someone asks you whether the information is authenticated, you have to assure them that the information is authenticated.
- If the question is NOT related to CSJM University, respond ONLY with:
  "I can only answer questions related to CSJM University."
- Do NOT answer general knowledge, general songs, movies, or any off-topic queries.
- Combine partial answers into one clear, concise response.
- Do not repeat information."""),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", """User Query: {user_query}


Partial Answers:
{combined_answers}

Instructions:
1. Answer ONLY from the context. Do not use outside knowledge.
2. If the answer is not found in context, respond exactly:
   "I'm sorry, this information is not available in my database.
   For more details, please contact us:
   📞 Phone: 8090803220
   🌐 Website: https://csjmu.ac.in/
   📧 Email: infocsjmu@csjmu.ac.in"
3. If answer IS found but incomplete, give the answer AND add at the end:
   "For more details: 🌐 https://csjmu.ac.in/ | 📞 8090803220"
4. Use markdown format where needed (tables, lists).
5. If user asked in Hindi or Hinglish, respond in Corresponding language.
6. Be polite, clear and concise.

Answer:""")
])


    
# ─────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────

async def rag_pipeline(user_query: str) -> str:

    # ── STEP 1 — Decompose ──────────────────
    print("=" * 60)
    print("STEP 1: QUERY DECOMPOSITION")
    print("=" * 60)
    print(f"Original Query: {user_query}\n")

    queries = await decompose_chain.ainvoke({"user_query": user_query})

    for i, q in enumerate(queries, 1):
        print(f"  Decomposed Query {i}: {q}")

    # ── STEP 2 — Refine ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: SUMMARY MATCH")
    print("=" * 60)

    refined_queries = await refine_all_queries(queries)

    for i, (query, refined) in enumerate(zip(queries, refined_queries), 1):
        print(f"\n  Decomposed Query {i} : {query}")
        print(f"  Matched Summary   {i} : {refined[:200]}...")

    # ── STEP 3 — Hybrid Retrieval + Router Parallel ──
    print("\n" + "=" * 60)
    print("STEP 3: HYBRID RETRIEVAL + ROUTER (PARALLEL)")
    print("=" * 60)

    # fetch_docs aur get_doc_id_by_intent parallel chalao
    fetch_results, router_results = await asyncio.gather(
        asyncio.gather(
            *[fetch_docs(query, refined)
              for query, refined in zip(queries, refined_queries)]
        ),
        asyncio.gather(
            *[get_doc_id_by_intent(query)
              for query in queries]
        )
    )

    # Router result aaya to use karo, nahi aaya to fetch_docs ka use karo
    final_results = []
    for i, (fetch_result, router_result) in enumerate(
        zip(fetch_results, router_results), 1
    ):
        if router_result is not None:
            print(f"\n  Query {i} → Router result used ✅")
            final_results.append(router_result)
        else:
            print(f"\n  Query {i} → Fetch docs result used ✅")
            final_results.append(fetch_result)

    retrieved_contexts = []
    for i, (query, (search_input, docs)) in enumerate(
        zip(queries, final_results), 1
    ):
        print(f"\n  --- Query {i} ---")
        print(f"  Decomposed Query : {query}")
        print(f"  Search Input     : {search_input[:150]}...")

        context = ""
        for j, doc in enumerate(docs, 1):
            original_content = doc.metadata.get('original_content', doc.page_content)
            print(f"\n  Document {j} original_content:")
            print(f"  {original_content[:150]}...")
            context += original_content + "\n\n"

        retrieved_contexts.append(context.strip())

    # ── STEP 4 — LLM Per Query ───────────────
    print("\n" + "=" * 60)
    print("STEP 4: LLM ANSWER PER DECOMPOSED QUERY")
    print("=" * 60)

    individual_answers = await asyncio.gather(
        *[get_answer(query, context)
          for query, context in zip(queries, retrieved_contexts)]
    )

    for i, (query, answer) in enumerate(zip(queries, individual_answers), 1):
        print(f"\n  Decomposed Query {i} : {query}")
        print(f"  LLM Answer       {i} : {answer}")

    # ── STEP 5 — Final Answer ────────────────
    print("\n" + "=" * 60)
    print("STEP 5: FINAL COMBINED ANSWER")
    print("=" * 60)

    combined_answers = "\n\n".join([
        f"Q: {query}\nA: {answer}"
        for query, answer in zip(queries, individual_answers)
    ])

    final_answer = await (
        final_prompt | llm | StrOutputParser()
    ).ainvoke({
        "chat_history" : history,
        "user_query": user_query,
        "combined_answers": combined_answers
    })

    history.extend([HumanMessage(content=user_query), AIMessage(content=final_answer)])

    print(f"\n  Original Query : {user_query}")
    print(f"\n  Final Answer   :\n  {final_answer}")
    print("=" * 60)

    return final_answer