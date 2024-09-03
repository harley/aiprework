import chainlit as cl
from openai import AsyncOpenAI
import os
import base64
import boto3


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


# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
bucket_name = os.getenv("S3_BUCKET_NAME")


def upload_to_s3(file_path, file_name):
    try:
        s3_client.upload_file(file_path, bucket_name, file_name)
        s3_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": file_name},
            ExpiresIn=3600,  # URL expires in 1 hour
        )
        return s3_url
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None


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
        image_urls = []
        for image in images:
            try:
                s3_url = upload_to_s3(image.path, os.path.basename(image.path))
                if s3_url:
                    # print(f"Debug: image loaded to {s3_url}")
                    image_urls.append(s3_url)
            except Exception as e:
                print(f"Error processing image {image.path}: {e}")

        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            message.content
                            if message.content
                            else "What's in these images?"
                        ),
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
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
