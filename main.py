from fastapi import FastAPI, Request
import logging

app = FastAPI()

logging.basicConfig(filename="app.log", level=logging.INFO)

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    chat_id = body.get("chat_id")
    messages = body.get("messages", [])

    response = {
        "message": None,
        "base_random_keys": None,
        "member_random_keys": None
    }

    if not messages:
        return {"error": "No messages provided"}

    last_message = messages[-1].get("content", "")
    logging.info(f"chat_id={chat_id}, message={last_message}")

    if last_message == "ping":
        response["message"] = "pong"
    elif last_message.startswith("return base random key:"):
        key = last_message.split(":", 1)[1].strip()
        response["base_random_keys"] = [key]
    elif last_message.startswith("return member random key:"):
        key = last_message.split(":", 1)[1].strip()
        response["member_random_keys"] = [key]

    return response
