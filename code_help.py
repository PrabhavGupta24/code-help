import os
from time import perf_counter
#from unidecode import unidecode

from gpt4all import GPT4All

import evadb

APP_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_FILE_PATH = os.path.join(APP_SOURCE_DIR, "code", "sample.txt")

#os.environ["OPENAI_KEY"] = "sk-..."

def ask_question(): #story_path: str
    llm = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")

    #path = os.path.dirname(os.getcwd())
    cursor = evadb.connect().cursor()


    option = int(
        input(
            "1. Input your own file, 2. Use sample file, 3. Don't use file: "
        )
    )

    using_file = option < 3
    file_path = DEFAULT_FILE_PATH
    if (using_file):
        if (option == 1):
            file_path = str(
                input(
                    "Enter your file path: "
                )
            )
        cursor.drop_table("CODEFILE", if_exists=True).execute()
        cursor.load(file_path, "CODEFILE", "document").execute()

        # cursor.query(
        #     """CREATE TABLE IF NOT EXISTS CODE_FILE;"""
        # ).execute()
        # try:
        #     with open(file_path, "r") as file:
        #         python_code = file.read()
        # except FileNotFoundError:
        #     print(f"File not found: {file_path}")
        # except Exception as e:
        #     print(f"An error occurred: {str(e)}")
        # i = 1
        # python_code = unidecode(python_code)
        # cursor.query(
        #     f"""INSERT INTO CODEFILE (id, data)
        #         VALUES ({i}, '{python_code}');"""
        # ).execute()

    context = cursor.query(f"""SELECT * FROM CODEFILE;""").df().values[0][3]
    if not using_file:
        question = input("Ask me a coding question: ")
        query = f"""Question : {question}"""
        full_response = llm.generate(query)
    else:
        question = input("Ask me a coding question: ")

        query = f"""
        If the context is not relevant, please answer the question by using your own knowledge about the topic.
        {context}
        Question : {question}"""

        full_response = llm.generate(query)
    print(full_response)


def main():
    ask_question()


if __name__ == "__main__":
    main()