# rem-RAGchain

Integrates [RAGchain](https://github.com/NomaDamas/RAGchain) and [rem](https://github.com/jasonjmcghee/rem).

`rem` is fascinating project, but don't have RAG feature yet. And of course, RAG with `rem` will be wonderful.
As `RAGchain` maker, I really want to use `rem` with RAG feature. So I quickly made this repo.

I just did a little experiment, and it works. However, answer quality is not good enough.
I'll have to get more data from `rem` and do experiment more pipeline.

## Install

1. Install `rem`. Only available on apple silicon.
2. Install requirements. ```pip install -r requirements.txt```
3. set `.env` file. There is a template file `.env.template`. You can use Dynamo Linker instead of Redis DB. Json Linker
   at RAGchain v0.2.4 is not stable, so it will not work properly.

## Usage

### Ingest rem data

You can ingest rem data with minute range. It ingests data past from minute range setting that you set.

```bash
    python3 ingest.py --db_path="<your/path/to/db.sqlite3>" --ingest_minutes=10
```

It will ingest past 10 minutes of `rem` data. It will be great you set this to crontab.

### Run RAGchain

You can talk about your `rem` record with RAGchain.

```bash
    python3 run_llm.py --query="<your query>"
```
