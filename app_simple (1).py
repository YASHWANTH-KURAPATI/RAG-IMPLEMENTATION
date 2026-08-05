import os
from functools import lru_cache

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Required top-level export for deployment
app = FastAPI(title="Gemini RAG Demo")
application = app
handler = app


TEXT = """The Internet is a global system of interconnected computer networks that uses
the Internet protocol suite (TCP/IP) to communicate between networks and devices.

The origins of the Internet date back to packet switching research commissioned by
the United States Department of Defense in the 1960s. The primary precursor network
was ARPANET, which initially connected academic and research networks.

The National Science Foundation Network (NSFNET) expanded networking during the
1980s. Commercialization in the mid-1990s helped the Internet expand worldwide.

Today the Internet supports the World Wide Web, email, cloud computing, video
conferencing, online gaming, social media, file sharing, commerce, education,
government, healthcare, and many other services."""


class Question(BaseModel):
    question: str


@lru_cache(maxsize=1)
def make_rag():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "No API key configured. Set the GEMINI_API_KEY environment variable on Vercel."
        )

    docs = [Document(page_content=TEXT)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    store = FAISS.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question only from the context below.
If the answer is not available, say: I don't know based on the provided document.

Context:
{context}

Question:
{question}

Answer:"""
    )

    def join_docs(items):
        return "\n\n".join(x.page_content for x in items)

    chain = (
        {"context": retriever | join_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_rag(question):
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    return make_rag().invoke(question)


def html_page(question="", answer="", error=""):
    # Deliberately plain/minimal UI, matching the user's reference.
    q = (
        question.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    a = (
        answer.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    e = (
        error.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    answer_block = f"<p><b>Answer:</b> {a}</p>" if answer else ""
    error_block = f"<p><b>Error:</b> {e}</p>" if error else ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Gemini RAG Demo</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: white;
            color: black;
            margin: 0;
        }}
        .box {{
            width: 620px;
            max-width: 90%;
            margin: 70px auto;
        }}
        h1 {{
            font-size: 34px;
            margin-bottom: 18px;
        }}
        .example {{
            font-family: monospace;
            white-space: pre-wrap;
            margin-bottom: 18px;
            line-height: 1.5;
        }}
        textarea {{
            width: 100%;
            height: 100px;
            padding: 8px;
            font-family: monospace;
            font-size: 14px;
            border: 1px solid #999;
            border-radius: 0;
            box-sizing: border-box;
        }}
        select, button {{
            margin-top: 10px;
            padding: 5px 10px;
            font-size: 14px;
        }}
        button {{
            margin-left: 5px;
        }}
        .output {{
            font-family: monospace;
            white-space: pre-wrap;
            margin-top: 18px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
<div class="box">
    <h1>🔴🟢 Gemini RAG Demo</h1>

    <div class="example">POST a question to /ask as JSON:

{{
  "question": "What were the origins of the Internet?",
  "mode": "rag" // or "gemini"
}}

Or try below:</div>

    <form method="post" action="/ask">
        <textarea name="question" placeholder="What were the origins of the Internet and what was its precursor network?" required>{q}</textarea>
        <br>
        <select name="mode">
            <option value="rag">Plain RAG</option>
        </select>
        <button type="submit">Ask</button>
    </form>

    <div class="output">{answer_block}{error_block}</div>

    <div class="output">Errors: "No API key configured. Set the GEMINI_API_KEY environment variable on Vercel."</div>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(html_page())


@app.post("/ask", response_class=HTMLResponse)
def ask_form(question: str = Form(...), mode: str = Form("rag")):
    try:
        answer = ask_rag(question)
        return HTMLResponse(html_page(question, answer))
    except Exception as exc:
        return HTMLResponse(html_page(question, error=str(exc)), status_code=500)


@app.post("/api/ask")
def ask_api(payload: Question):
    try:
        return {"answer": ask_rag(payload.question)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
