from fastapi import FastAPI
from pydantic import BaseModel
from  fastapi.middleware.cors import CORSMiddleware

from nl2sql import (
    nl_to_sql,
    execute_query,
    generate_natural_answer
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    sql = nl_to_sql(pertanyaan)

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