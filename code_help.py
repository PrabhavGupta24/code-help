import os
from time import perf_counter
from unidecode import unidecode
import warnings

from gpt4all import GPT4All

import evadb

APP_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_FILE_PATH = os.path.join(APP_SOURCE_DIR, "code", "sample.txt")
warnings.filterwarnings("ignore")


def ask_question(): #story_path: str
    #llm = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
    cursor = evadb.connect().cursor()

    # option = int(
    #     input(
    #         "(Enter a Number) 1. Input your own file, 2. Use sample file, 3. Don't use file: "
    #     )
    # )

    # using_file = option < 3

    option = 2
    using_file = True
    file_path = DEFAULT_FILE_PATH
    if (using_file):
        if (option == 1):
            file_path = str(
                input(
                    "Enter your file path: "
                )
            )
        
        try:
            with open(file_path, "r") as file:
                python_code = file.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        print(type(python_code))
        python_code = unidecode(python_code)
        print(type(python_code))

        code_table = "CODEFILE"
        code_embed_table = "CodeEmbeddings"
        index_table = "IndexTable"

        code_embed_function_query = f"""CREATE UDF IF NOT EXISTS EmbedExtractor
            IMPL  './code_embedding_extractor.py';
            """

        cursor.query("DROP UDF IF EXISTS EmbedExtractor;").execute()
        cursor.query(code_embed_function_query).execute()

        cursor.query(f"DROP TABLE IF EXISTS {code_table};").execute()
        cursor.query(f"DROP TABLE IF EXISTS {code_embed_table};").execute()

        print("set up done")

        cursor.query(f"CREATE TABLE {code_table} (id INTEGER, data TEXT(4096));").execute()

        print("table made")

        i = 0
        cursor.query(
            f"""INSERT INTO {code_table} (id, data)
                VALUES ({i}, '{python_code}');"""
        ).execute()
        print("File Loaded")



        cursor.query(
            f"""CREATE TABLE {code_embed_table} AS
            SELECT EmbedExtractor(data), data FROM {code_table};"""
        ).execute()


        # Test Code
        # context = cursor.query(f"SELECT * FROM {code_table};").df()
        # print(type(context["codefile.data"][0]))



        print("Code Embeddings Generated")
        return
        
   
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
    cursor = evadb.connect().cursor()
    ask_question()


if __name__ == "__main__":
    main()