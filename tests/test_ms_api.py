from openai import OpenAI

client = OpenAI(api_key="EMPTY",
                base_url="http://0.0.0.0:20000/v1")
model = "mindspore-api"

# create a chat completion
completion = client.chat.completions.create(model=model,
                                            messages=[{"role": "user", "content": "Hello! What is your name?"}])
# print the completion
print(completion.choices[0].message.content)
