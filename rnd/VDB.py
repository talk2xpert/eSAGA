import numpy as np
import faiss
import json

# Sample JSON data
data = [
    {
        "id": "1",
        "title": "Introduction to Spring Boot",
        "content": "Spring Boot is an open-source framework that allows you to build production-ready applications quickly and easily.",
        "tags": ["spring", "java", "framework"],
        "date": "2023-01-01"
    },
    {
        "id": "2",
        "title": "Getting Started with PostgreSQL",
        "content": "PostgreSQL is a powerful, open-source object-relational database system with over 30 years of active development.",
        "tags": ["database", "sql", "postgresql"],
        "date": "2023-02-01"
    },
    {
        "id": "3",
        "title": "Dynamic Queries in Spring Boot",
        "content": "This guide covers how to create dynamic SQL queries in a Spring Boot application using JPA and Criteria API.",
        "tags": ["spring boot", "dynamic queries", "jpa", "sql"],
        "date": "2023-03-01"
    }
]

# Generate random vectors for each document (assuming 128-dimensional vectors)
np.random.seed(42)
dimension = 128
vectors = np.random.random((len(data), dimension)).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Save metadata for retrieving original documents
id_to_doc = {i: doc for i, doc in enumerate(data)}

def query_vector(query_vector, k=3):
    distances, indices = index.search(query_vector, k)
    results = [id_to_doc[idx] for idx in indices[0]]
    return results

# Generate a random query vector (128-dimensional)
query_vec = np.random.random((1, dimension)).astype('float32')

# Perform the search
results = query_vector(query_vec)

# Print the results
print(json.dumps(results, indent=4))
