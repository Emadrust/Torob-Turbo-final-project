from fastapi import FastAPI, Request
import logging

app = FastAPI()

logging.basicConfig(filename="app.log", level=logging.INFO)

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    chat_id = body.get("chat_id")
    messages = body.get("messages", [])
    response = {}

    if not messages:
        return {"error": "No messages provided"}

    last_message = messages[-1].get("content", "")

    logging.info(f"chat_id={chat_id}, message={last_message}")

    if last_message == "ping":
        response["message"] = "pong"
    elif last_message.startswith("return base random key:"):
        key = last_message.split(":", 1)[1]
        response["base_random_keys"] = [key]
    elif last_message.startswith("return member random key:"):
        key = last_message.split(":", 1)[1]
        response["member_random_keys"] = [key]
    else:
        response["message"] = "unrecognized command"

    return response
