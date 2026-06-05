from llm_client import create_zhipu_client
from llm_client.llm_factory import create_step_client
from llm_client.models import Message

client = create_step_client()
resp = client.completion(messages=[Message(role="user", content="hi")])
print(resp.content)
