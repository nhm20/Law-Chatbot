import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl
import speech_recognition as sr

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Paths
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

async def process_and_send_response(res):
    answer = res["result"]
    sources = res["source_documents"]
    
    if sources:
        source_texts = [
            f"Document {i+1}: Page {doc.metadata.get('page', 'N/A')}" 
            for i, doc in enumerate(sources)
        ]
        answer += f"\nSources: {', '.join(source_texts)}"
    else:
        answer += "\nNo sources found"
    
    await cl.Message(content=answer).send()

async def handle_voice_input():
    # Show listening message
    listening_msg = None
    try:
        listening_msg = await cl.Message(content="🔊 Listening... Speak now").send()
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            
            text = recognizer.recognize_google(audio)
            await cl.Message(content="✅ Processing your voice input...").send()
            await cl.Message(content=f"🎙️ You said: {text}").send()
            return text
                
    except sr.UnknownValueError:
        await cl.Message(content="❌ Could not understand audio").send()
    except sr.RequestError as e:
        await cl.Message(content=f"❌ Speech service error: {e}").send()
    except Exception as e:
        await cl.Message(content=f"❌ Microphone error: {str(e)}").send()
    finally:
        if listening_msg:
            try:
                await listening_msg.remove()
            except:
                pass
    return None

@cl.on_chat_start
async def start():
    # Initialize QA chain
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    
    # Send welcome message with voice button
    await cl.Message(
        content="""
**Welcome to Law Bot!**  

You can:  
1. Type your legal question  
2. Click the button below to speak  
3. Or type 'voice' to use microphone  
""",
        actions=[
            cl.Action(name="voice_action", value="voice", label="🎤 Speak Now")
        ]
    ).send()

@cl.action_callback("voice_action")
async def on_voice_action(action: cl.Action):
    text = await handle_voice_input()
    if text:
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        res = await chain.ainvoke(text, callbacks=[cb])
        await process_and_send_response(res)

@cl.on_message
async def main(message: cl.Message):
    # Check for voice command
    if message.content.lower().strip() == "voice":
        text = await handle_voice_input()
        if not text:
            return
        
        # Process voice input through QA chain
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        res = await chain.ainvoke(text, callbacks=[cb])
        await process_and_send_response(res)
        return
        
    # Process normal text input
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])
        await process_and_send_response(res)
    except Exception as e:
        logging.error(f"Error: {e}")
        await cl.Message(content="❌ An error occurred. Please try again.").send() 