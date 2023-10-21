import os
from unidecode import unidecode
import warnings

from gpt4all import GPT4All

import evadb

APP_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_FILE_PATH = os.path.join(APP_SOURCE_DIR, "code", "sample.txt")
warnings.filterwarnings("ignore")


def ask_question():
    llm = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
    cursor = evadb.connect().cursor()

    option = int(
        input(
            "(Enter a Number) 1. Input your own file, 2. Use sample file, 3. Don't use file: "
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
        try:
            with open(file_path, "r") as file:
                python_code = file.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        python_code = unidecode(python_code)

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

        cursor.query("DROP UDF IF EXISTS Similarity;").execute()
        Similarity_function_query = """CREATE UDF Similarity
                        INPUT (Frame_Array_Open NDARRAY UINT8(3, ANYDIM, ANYDIM),
                            Frame_Array_Base NDARRAY UINT8(3, ANYDIM, ANYDIM),
                            Feature_Extractor_Name TEXT(100))
                        OUTPUT (distance FLOAT(32, 7))
                        TYPE NdarrayFunction
                        IMPL './similarity.py'"""
        
        cursor.query(Similarity_function_query).execute()
        print("Set Up Done")

        cursor.query(f"CREATE TABLE {code_table} (id INTEGER, data TEXT(4096));").execute()
        print("Table Made")

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
        print("Code Embeddings Generated")

        cursor.query(
            f"CREATE INDEX {index_table} ON {code_embed_table} (embeddings) USING QDRANT;"
        ).execute()
        print("Index Created")

        # Test Code
        # context = cursor.query(f"SELECT * FROM {code_table};").df()
        # print(type(context["codefile.data"][0]))

        
    question = input("Ask me a coding question: ")
    question = unidecode(question)
    if not using_file:
        query = f"""Question : {question}"""
        full_response = llm.generate(query)
    else:
        batch = cursor.query(
            f"""SELECT data FROM {code_embed_table}
            ORDER BY Similarity(EmbedExtractor('{question}'),embeddings)
            LIMIT 5;"""
        ).execute()
        print("Batch Created")

        context_list = []
        for i in range(len(batch)):
            context_list.append(batch.frames[f"{code_embed_table.lower()}.data"][i])
        context = "\n".join(context_list)
        print("Context Created")
        print(context)

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