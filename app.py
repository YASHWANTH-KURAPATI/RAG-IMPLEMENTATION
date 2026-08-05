import os
import html
from functools import lru_cache

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
# These top-level names make the application easy to detect
# on common Python/FastAPI deployment platforms.
app = FastAPI(title="Gemini RAG Demo")
application = app
handler = app


# ---------------------------------------------------------
# Demo document used by RAG
# ---------------------------------------------------------
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


def get_google_api_key() -> str:
    """
    Accept either environment-variable name.

    Recommended on Vercel:
        GOOGLE_API_KEY = your Gemini API key

    GEMINI_API_KEY is also supported so the application does not fail
    simply because a different common variable name was used.
    """
    api_key = (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )

    if not api_key:
        raise RuntimeError(
            "Gemini API key is not configured. "
            "In Vercel go to Project Settings > Environment Variables, "
            "add GOOGLE_API_KEY with your Gemini API key, then redeploy."
        )

    return api_key


@lru_cache(maxsize=1)
def make_rag():
    api_key = get_google_api_key()

    # 1. Convert the source text into LangChain documents.
    docs = [Document(page_content=TEXT)]

    # 2. Split the document into smaller chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    # 3. Create Gemini embeddings and store them in FAISS.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=api_key,
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    # 4. Gemini chat model.
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.2,
    )

    # 5. RAG prompt.
    prompt = ChatPromptTemplate.from_template(
        """Answer the question only from the context below.
If the answer is not available in the context, say exactly:
I don't know based on the provided document.

Context:
{context}

Question:
{question}

Answer:"""
    )

    def join_docs(items):
        return "\n\n".join(doc.page_content for doc in items)

    # 6. LCEL RAG chain.
    chain = (
        {
            "context": retriever | join_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask_rag(question: str) -> str:
    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    return make_rag().invoke(question)


def html_page(question: str = "", answer: str = "", error: str = "") -> str:
    q = html.escape(question, quote=True)
    a = html.escape(answer)
    e = html.escape(error)

    answer_block = (
        f'<div class="result"><b>Answer:</b><br>{a}</div>'
        if answer
        else ""
    )

    error_block = (
        f'<div class="error"><b>Error:</b><br>{e}</div>'
        if error
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini RAG Demo</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #ffffff;
            color: #111111;
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
            border: 1px solid #999999;
            border-radius: 0;
            box-sizing: border-box;
            resize: vertical;
        }}

        select,
        button {{
            margin-top: 10px;
            padding: 6px 10px;
            font-size: 14px;
        }}

        button {{
            margin-left: 5px;
            cursor: pointer;
        }}

        .result,
        .error {{
            font-family: monospace;
            white-space: pre-wrap;
            margin-top: 20px;
            line-height: 1.5;
        }}

        .error {{
            color: #b00020;
        }}

        .note {{
            font-family: monospace;
            margin-top: 24px;
            line-height: 1.5;
            color: #444444;
        }}
    </style>
</head>

<body>
    <div class="box">
        <h1>🔴🟢 Gemini RAG Demo</h1>

        <div class="example">POST JSON to /api/ask:

{{
  "question": "What were the origins of the Internet?"
}}

Or try below:</div>

        <form method="post" action="/ask">
            <textarea
                name="question"
                placeholder="What were the origins of the Internet and what was its precursor network?"
                required>{q}</textarea>
            <br>

            <select name="mode">
                <option value="rag">Plain RAG</option>
            </select>

            <button type="submit">Ask</button>
        </form>

        {answer_block}
        {error_block}

        <div class="note">
            API endpoint: POST /api/ask<br>
            Health check: GET /health
        </div>
    </div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=html_page())


@app.post("/ask", response_class=HTMLResponse)
def ask_form(
    question: str = Form(...),
    mode: str = Form("rag"),
):
    try:
        if mode != "rag":
            raise ValueError("Only Plain RAG mode is currently supported.")

        answer = ask_rag(question)
        return HTMLResponse(content=html_page(question=question, answer=answer))

    except ValueError as exc:
        return HTMLResponse(
            content=html_page(question=question, error=str(exc)),
            status_code=400,
        )

    except Exception as exc:
        return HTMLResponse(
            content=html_page(question=question, error=str(exc)),
            status_code=500,
        )


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
    return {
        "status": "ok",
        "api_key_configured": bool(
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
