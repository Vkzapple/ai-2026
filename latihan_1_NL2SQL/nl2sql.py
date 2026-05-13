from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import pandas as pd

load_dotenv()

client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)

SCHEMA = '''
Table: students

Columns:
- student_id (VARCHAR)
- gender (VARCHAR)
- age (INT)
- study_hours (FLOAT)
- attendance_pct (FLOAT)
- sleep_hours (FLOAT)
- final_score (FLOAT)
- passed (INT: 1=lulus, 0=tidak lulus)
'''


def nl_to_sql(pertanyaan: str) -> str:

    prompt = f'''
You are a MySQL Expert.

Database Schema:
{SCHEMA}

Rules:
1. Only output SQL query
2. No markdown
3. No explanation
4. Use LIMIT 10 for display queries without specific amount
5. Use COUNT(*) for total/jumlah questions
6. Use proper MySQL syntax
7. Understand Indonesian and English questions

Examples:

Q: tampilkan siswa yang lulus
SQL:
SELECT * FROM students WHERE passed = 1 LIMIT 10;

Q: jumlah siswa perempuan
SQL:
SELECT COUNT(*) FROM students WHERE gender = 'Female';

Q: rata-rata nilai siswa laki-laki
SQL:
SELECT AVG(final_score) FROM students WHERE gender = 'Male';

Q: siswa dengan nilai tertinggi
SQL:
SELECT * FROM students ORDER BY final_score DESC LIMIT 1;

User Question:
{pertanyaan}

SQL:
'''

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.1,
        max_tokens=150
    )

    sql = response.choices[0].message.content.strip()

    # bersihin kalau model masih bandel
    sql = sql.replace('```sql', '')
    sql = sql.replace('```', '')
    sql = sql.replace('SQL:', '')
    sql = sql.strip()

    return sql


def execute_query(sql: str) -> dict:

    try:

        engine = create_engine(
            f'mysql+pymysql://{os.getenv("DB_USER")}:'
            f'{os.getenv("DB_PASSWORD")}'
            f'@{os.getenv("DB_HOST")}/'
            f'{os.getenv("DB_NAME")}'
        )

        with engine.connect() as conn:

            result = pd.read_sql(
                text(sql),
                conn
            )

            return {
                'status': 'success',
                'data': result.to_dict(orient='records'),
                'total': len(result)
            }

    except Exception as e:

        return {
            'status': 'error',
            'message': str(e)
        }


def generate_natural_answer(
    pertanyaan: str,
    data
) -> str:

    prompt = f'''
You are an AI Data Analyst.

User question:
{pertanyaan}

SQL Result:
{data}

Task:
- Answer naturally like ChatGPT
- Use Indonesian
- Make response short, clear, friendly
- Explain the result from the data
- Do not mention SQL
- Do not output JSON
'''

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.3,
        max_tokens=200
    )

    jawaban = response.choices[0].message.content.strip()

    return jawaban