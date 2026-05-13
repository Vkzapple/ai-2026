from fastapi import FastAPI
from pydantic import BaseModel

from nl2sql import (
    nl_to_sql,
    execute_query,
    generate_natural_answer
)

app = FastAPI()


class Question(BaseModel):
    pertanyaan: str


@app.get('/')
def home():

    return {
        'message': 'NL2SQL API berjalan'
    }


@app.post('/ask')
def ask(data: Question):

    pertanyaan = data.pertanyaan

    # AI generate SQL
    sql = nl_to_sql(pertanyaan)

    # execute SQL
    hasil = execute_query(sql)

    if hasil['status'] == 'error':

        return {
            'status': 'error',
            'message': hasil['message']
        }

    jawaban = generate_natural_answer(
        pertanyaan,
        hasil['data']
    )

    return {
        'status': 'success',
        'pertanyaan': pertanyaan,
        'jawaban': jawaban,
        'data': hasil['data']
    }