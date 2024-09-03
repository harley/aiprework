import chainlit as cl
from openai import AsyncOpenAI
import os
import base64


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint_url = "https://api.openai.com/v1"
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint_url)
    model_kwargs = {"model": "chatgpt-4o-latest", "temperature": 0.5, "max_tokens": 500}
    return client, model_kwargs


def get_runpod_client():
    api_key = os.getenv("RUNPOD_API_KEY")
    runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")
    endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint_url)
    model_kwargs = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "temperature": 0.5,
        "max_tokens": 500,
    }
    return client, model_kwargs


use_openai = True  # Set this to False to use RunPod endpoint

if use_openai:
    client, model_kwargs = get_openai_client()
else:
    client, model_kwargs = get_runpod_client()


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    # parse images
    images = (
        [file for file in message.elements if "image" in file.mime]
        if message.elements
        else []
    )

    if images:
        # read the first image and encode to base64
        first_image = images[0]
        with open(first_image.path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            message.content
                            if message.content
                            else "What's in this image?"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
    else:
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
