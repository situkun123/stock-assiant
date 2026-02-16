import chainlit as cl
from langchain_community.callbacks import get_openai_callback
from agent import create_financial_agent, get_cached_companies


@cl.on_chat_start
async def start():
    """Initialize the agent when a new chat session starts."""
    app = create_financial_agent()
    cl.user_session.set("agent", app)
    await cl.Message(
        content="👋 Welcome to the Financial Analysis Assistant! Ask me about stock comparisons, company info, or financial metrics. Try: 'Compare TSLA and F' or 'What's the P/E ratio of AAPL?'"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from the user."""
    # Get the agent from session
    app = cl.user_session.get("agent")
    
    # Show a loading message
    msg = cl.Message(content="")
    await msg.send()
    
    # Run the agent with cost tracking
    with get_openai_callback() as cb:
        result = app.invoke({"messages": [("user", message.content)]})
        final_message = result["messages"][-1].content
        
        # Update the message with the result
        msg.content = final_message
        await msg.update()
        
        # Get cached companies
        cached = get_cached_companies()
        
        # Send cost information as a separate message
        cost_info = f"""
            📊 **Usage Statistics:**
            - Total Tokens: {cb.total_tokens:,}
            - Prompt Tokens: {cb.prompt_tokens:,}
            - Completion Tokens: {cb.completion_tokens:,}
            - Total Cost: ${cb.total_cost:.6f} USD
            - API Calls: {cb.successful_requests}
            - Cached Companies: {', '.join(cached) if cached else 'None'}
            """
        
        await cl.Message(
            content=cost_info,
            author="System"
        ).send()


if __name__ == "__main__":
    # This will be handled by the chainlit run command
    pass