"""Jack — S-1. FastAPI hub entrypoint."""


def main() -> None:
    import uvicorn
    uvicorn.run("gpushare.server.app:app", host="0.0.0.0", port=8000, reload=True)
