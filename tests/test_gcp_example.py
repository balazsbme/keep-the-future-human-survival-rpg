from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()
response = client.models.embed_content(
    model="text-embedding-005",
    contents=[
        "How do I get a driver's license/learner's permit?",
        "How long is my driver's license valid for?",
        "Driver's knowledge test study guide",
    ],
    config=EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",  # Optional
        output_dimensionality=768,  # Optional
        title="Driver's License",  # Optional
    ),
)
print(response)
# Example response:
# embeddings=[ContentEmbedding(values=[-0.06302902102470398, 0.00928034819662571, 0.014716853387653828, -0.028747491538524628, ... ],
# statistics=ContentEmbeddingStatistics(truncated=False, token_count=13.0))]
# metadata=EmbedContentMetadata(billable_character_count=112)