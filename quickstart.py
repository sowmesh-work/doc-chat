import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "What should I search for to find the latest developments in renewable energy?",
        }
    ],
)

for block in message.content:
    if block.type == "text":
        print(block.text)
print(message.content)
print(message.stop_reason)
print(message.usage)