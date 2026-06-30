FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "langgraph-cli[inmem]" \
    langchain-openai \
    langchain-postgres \
    "psycopg[binary]"

COPY . /deps/__outer_rag_agent
RUN pip install --no-cache-dir -e /deps/__outer_rag_agent

WORKDIR /deps/__outer_rag_agent

EXPOSE 8123

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123", "--no-browser"]
