# Chat with your own data

This framework provides a straightforward chatbot interface that allows you to interact with your data using a Language Learning Model (LLM). It offers the flexibility to connect with various LLM models, such as OpenAI’s GPT-4. Additionally, it can interface with different types of Vector Databases, including Redis and Azure Cognitive Search. By default, it utilizes a local FAISS Database, enabling the chatbot to operate independently without the need for an external Vector Database.

# Getting Started

1. Create a virtual enviroment. The solution is tested with Python 3.10.
    ```
    conda create -n openai-qna python=3.10
    conda activate openai-qna
    ```

2. Clone the repo and install python dependencies:
    ```
    git clone https://github.com/andyhemsft/openai-qna.git
    cd openai-qna
    pip install -r requirements.txt
    ```

3. Run the following command in the root directory of your local repository:
   ```
   flask run
   ```

4. Access the url localhost:5000/ui. You should see your chatbot running:
   ![image](https://github.com/andyhemsft/openai-qna/assets/64599697/05d3858c-5dc4-4c92-bc60-447fc2cb1d6a)

# Prepare your data

TBD
