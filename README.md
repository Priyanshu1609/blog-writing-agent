# Blog Agent

## Quickstart

- Install: `pip install -e .`
- Run locally: `python -m main --topic "Your topic"`
- Run news roundup: `python -m main --news --topic "Latest tech news"`
- API: `uvicorn api.app:app_api --reload`

## Environment

Set the following env vars (or use `.env`):

- OPENAI_API_KEY
- TAVILY_API_KEY (optional)
- GOOGLE_API_KEY (optional, for images)

Outputs are saved under `outputs/`.
# blog-writing-agent
# blog-writing-agent
