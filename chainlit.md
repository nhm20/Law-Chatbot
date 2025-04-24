import chainlit as cl

@cl.on_chat_start
async def start(): # Initialize your QA chain if needed # chain = qa_bot() # cl.user_session.set("chain", chain)
pass # No welcome message, just start the chat

@cl.on_message
async def main(message: cl.Message): # Your full QA processing logic here
await cl.Message(content=f"Processing: {message.content}").send()
