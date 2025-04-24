import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl

# Set up logging for debugging
logging.basicConfig(level=logging.ERROR)  # Only log errors

# Paths
DB_FAISS_PATH = 'vectorstore/db_faiss'  # FAISS vectorstore path

# Custom prompt template for the QA chain
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Function to set the custom prompt for the QA model
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Function to create the retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    # Use a smaller `k` value to limit the number of documents retrieved
    retriever = db.as_retriever(search_kwargs={'k': 2})  # Retrieve only 2 documents
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Function to load the language model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Function to split text into smaller chunks
def split_text_into_chunks(text, max_tokens=500):
    # Split text into words
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Approximate token count (1 word ≈ 1.3 tokens)
        word_length = len(word) + 1  # Add 1 for the space
        if current_length + word_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to initialize the QA bot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Function to return the final result for a user query
def final_result(query):
    try:
        qa_result = qa_bot()
        response = qa_result({'query': query})
        
        # Extract examples from the source documents
        examples = []
        for doc in response['source_documents']:
            if 'tags' in doc.metadata and 'example' in doc.metadata['tags']:
                # Split long documents into smaller chunks
                chunks = split_text_into_chunks(doc.page_content, max_tokens=500)
                examples.extend(chunks)
        
        # Format the response to include examples
        if examples:
            response['result'] += "\n\nExamples:\n" + "\n\n".join(examples)
        
        return response
    except Exception as e:
        logging.error(f"Error in final_result: {e}")
        return {"result": "An error occurred while processing your query.", "source_documents": []}

# Chainlit chat starting event
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Law Bot. What is your query?"
    await msg.update()

    # Store the chain in the user session
    cl.user_session.set("chain", chain)

# Chainlit message event for processing user messages
@cl.on_message
async def main(message: cl.Message):
    # Get the chain from the user session
    chain = cl.user_session.get("chain")
    
    # Callback handler for streaming the final answer
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    try:
        # Check if the query is too long
        if len(message.content.split()) > 500:  # Approximate token limit
            await cl.Message(content="Your query is too long. Please shorten it and try again.").send()
            return
        
        # Call the chain with the user's message
        res = await chain.ainvoke(message.content, callbacks=[cb])  # Updated method
        
        # Get the answer and the source documents
        answer = res["result"]
        sources = res["source_documents"]

        # Format the response to include the sources if available
        if sources:
            source_texts = [f"Document {i+1}: Page {doc.metadata.get('page', 'N/A')}" for i, doc in enumerate(sources)]
            answer += f"\nSources: {', '.join(source_texts)}"
        else:
            answer += "\nNo sources found"

        # Send the response back to the user
        await cl.Message(content=answer).send()
    except Exception as e:
        logging.error(f"Error during processing message: {e}")
        await cl.Message(content="An error occurred while processing your request. Please try again later.").send()