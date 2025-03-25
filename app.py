from fastapi import FastAPI
from pydantic import BaseModel
import re
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Load Model & Tokenizer
model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_prompt(user_question, prompt_template, table_metadata_string):
    return prompt_template.format(user_question=user_question, table_metadata_string=table_metadata_string)

def run_inference(question):
    prompt_template = """Task
    Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

    Instructions
    If you cannot answer the question with the available database schema, return 'I do not know'

    Database Schema
    The query will run on a database with the following schema: {table_metadata_string}

    Answer
    Given the database schema, here is the SQL query that answers [QUESTION]{user_question}[/QUESTION] [SQL]"""

    metadata = """CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        first_name VARCHAR(255),
        last_name VARCHAR(255),
        phone_number VARCHAR(50),
        active BOOLEAN DEFAULT TRUE
    );

    CREATE TABLE posts (
        id INTEGER PRIMARY KEY,
        title VARCHAR(255),
        description TEXT,
        user_id INTEGER NOT NULL,
        visibility VARCHAR(50),
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE comments (
        id INTEGER PRIMARY KEY,
        content TEXT,
        post_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        FOREIGN KEY (post_id) REFERENCES posts(id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE likes (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        likeable_type VARCHAR(50) NOT NULL,
        likeable_id INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );"""

    prompt = generate_prompt(question, prompt_template, metadata)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False,
        num_beams=5,
    )

    generated_text = pipe(prompt, num_return_sequences=1)[0]["generated_text"]

    # Extract SQL query
    sql_pattern = r"(SELECT .*?;)"
    match = re.search(sql_pattern, generated_text, re.DOTALL | re.IGNORECASE)
    generated_query = match.group(1) if match else "I do not know;"

    return generated_query

@app.post("/generate_sql")
def generate_sql(request: QueryRequest):
    sql_query = run_inference(request.question)
    return {"sql_query": sql_query}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
