import chainlit as cl
from openai import AsyncOpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

endpoint_url = "https://api.openai.com/v1"
client = AsyncOpenAI(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {"model": "chatgpt-4o-latest", "temperature": 0.5, "max_tokens": 500}


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(
        messages=history,
        stream=True,
        **model_kwargs,
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    ai_response = {"role": "assistant", "content": response_message.content}
    history.append(ai_response)
    cl.user_session.set("history", history)
