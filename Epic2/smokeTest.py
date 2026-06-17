import os

from openai import OpenAI

# Required: set NVIDIA_API_KEY in your environment before running this smoke test.
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=os.environ["NVIDIA_API_KEY"],
)

completion = client.chat.completions.create(
  model="openai/gpt-oss-20b",
  messages=[{"role":"user","content":"explain quantum computing in 100 words"}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=False
)

reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
if reasoning:
  print(reasoning)
print(completion.choices[0].message.content)