"""Prompt format shared by evaluation and training, so the student trains on exactly the format it is tested on."""

SYSTEM_PROMPT = (
    "You are an expert SQL analyst. You write a single DuckDB SQL query that answers the "
    "user's question against the given schema. Return exactly the requested output columns, "
    "in the requested order. Reply with only the SQL query inside a ```sql code block."
)


def build_prompt(schema: str, question: str) -> str:
    return f"Schema:\n\n{schema}\n\nQuestion:\n{question}"
