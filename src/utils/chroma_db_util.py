import chromadb
from torch import ge

# Step 1. Creata a client
chroma_client = chromadb.Client()

# Step 2. Create a collection: Where embeddings, documents and any additional metadata will be stored.
collection = chroma_client.create_collection(name="my_collection")

# Step 3. Add some text documents: Chromadb will handle embedding and indexing automatically.
# Must provide unique IDs for documents.

# Document 1: A brief note on the importance of daily habits.
document_1 = """
The Power of Small Actions

Consistency, not intensity, is the key to achieving your long-term goals. Focus on developing small, manageable daily habits. Whether it's reading for ten minutes, taking a short walk, or spending a few moments reflecting, these tiny actions compound over time into massive results. Start small, stay consistent, and watch the momentum build.
"""

# Document 2: A definition of a technical concept (The Fibonacci Sequence).
document_2 = """
Fibonacci Sequence Definition

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. The sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, and so on. This pattern appears frequently in nature, from the spirals of a sunflower head to the branching of trees.
"""

# Document 3: A short, hypothetical project requirement summary.
document_3 = """
Project Genesis: Core Requirements

1. Develop a user-friendly authentication system (login/signup).
2. Implement real-time data synchronization using a database.
3. Ensure the mobile view is fully responsive and optimized for touch input.
4. All services must maintain 99.9% uptime and pass all security audits.
"""

# Step 4: Add the documents to the collection with the ids
import uuid

def generate_random_id():
    """
    Generates a random UUID (Universally Unique Identifier).
    Example Usage:
    >>> random_id = generate_random_id()
    >>> print(f"Generated random ID: {random_id}")
    """
    return str(uuid.uuid4())
    
collection.add(
    ids = [generate_random_id(), generate_random_id(), generate_random_id()],
    documents=[document_1, document_2, document_3]
  )


# Step 5: Query the result

results = collection.query(
    query_texts=["About a project"],
    n_results=2
)

print("Here is the first result ✅")
for k,v in results.items():
    print(f"Key:{k}\t\t{v}", end="\n\n")


print("".join(["=" for x in range(1, 10)]))

results = collection.query(
    query_texts=["Goals"],
    n_results=2
)

print("Here is the second result ✅")
for k,v in results.items():
    print(f"Key:{k}\t\t{v}", end="\n\n")