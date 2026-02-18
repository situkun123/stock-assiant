import os
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.database import Logger
from backend.tools import (
    correct_period_parameter,
    get_company_info,
    get_financial_statements,
    get_stock_history,
    search_stock_symbol,
    validate_stock_symbol,
)

root_dir = Path(__file__).resolve().parent.parent

# 2. Add that root directory to sys.path
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

load_dotenv()

# ============================================================================
# STATE & GRAPH NODES 
# ============================================================================
# Using agent state to dynamically route between agent and tools based on tool calls in messages. This allows the agent to decide when to call tools and when to end the conversation, while also preventing infinite loops with a max tool call limit.
class AgentState(TypedDict):
    """State object that flows through the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_plan(state: AgentState, model, tools):
    """First step: ask the LLM to produce a step-by-step plan before using any tools."""
    tool_names = ", ".join(t.name for t in tools)
    planning_prompt = (
        f"Before taking any action, produce a concise numbered plan for how you will "
        f"answer the user's question using only these tools: {tool_names}. "
        f"Do NOT call any tools yet — only output the plan."
        f"Note if using get_stock_history tool, Always check any time related parameters with correct_period_parameter tool first to ensure they are valid before calling get_stock_history."
    )
    planning_messages = state["messages"] + [SystemMessage(content=planning_prompt)]
    plan = model.invoke(planning_messages)
    # Inject the plan as a SystemMessage so the agent node sees it as context
    return {"messages": [SystemMessage(content=f"Plan:\n{plan.content}")]}


def call_model(state: AgentState, model, tools):
    """Call LLM with tool binding to decide next action."""
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}



def should_continue(state: AgentState):
    """Route to tools if needed, otherwise end. Stops after 50 total tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        total_calls = sum(
            len(m.tool_calls)
            for m in state["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        )
        # limit total tool calls to 50 for one conversation to prevent infinite loops
        if total_calls >= 50:
            return "end"
        return "tools"
    return "end"


# ============================================================================
# AGENT CREATION
# ============================================================================
def create_financial_agent():
    """Create financial analysis agent using LangGraph StateGraph.
    START → planner → agent → tools → agent → ... → END"""
    tools = [validate_stock_symbol, 
             get_company_info, 
             get_stock_history, 
             get_financial_statements, 
             correct_period_parameter,
             search_stock_symbol]

    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", lambda state: create_plan(state, model, tools))
    workflow.add_node("agent", lambda state: call_model(state, model, tools))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END
    })
    workflow.add_edge("tools", "agent")

    db_path = Path(os.getenv("CHECKPOINTS_DB", root_dir / "data" / "checkpoints.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def run_financial_agent(app, user_query: str, thread_id: str = "default", enable_logging: bool = True):
    """Execute agent and return response with cost breakdown."""

    system_prompt = """
                You are a financial analysis assistant. Your role is to:
                - Analyze stock data and financial statements objectively
                - Provide clear, data-driven insights
                - Use available tools to gather accurate information
                - Always cite your data sources

                 When working with stock symbols:
                1. If a user provides a symbol, validate it using validate_stock_symbol first
                2. If validation fails, try search_stock_symbol with the company name
                3. If you get a company name instead of symbol, use search_stock_symbol to find the correct ticker
                4. Always confirm the symbol is valid before fetching data

                When working with time periods:
                5. Valid periods for get_stock_history are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max before calling get_stock_history.
                6. If the user requests a period NOT in that list (e.g. '1w', '2w', '3m', 'week', 'month', 'year'),
                   call correct_period_parameter FIRST to get the valid equivalent, then pass it to get_stock_history"""

    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    config = {"configurable": {"thread_id": thread_id}}

    # getting the API cost info
    with get_openai_callback() as cb:
        result = app.invoke({"messages": initial_messages}, config=config)
        tool_calls = 0
        tools_used = {}

        for message in result["messages"]:
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls += len(message.tool_calls)

                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "Unknown")
                    tools_used[tool_name] = tools_used.get(tool_name, 0) + 1

        response_content = result["messages"][-1].content
        metadata = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": round(cb.total_cost, 6),
            "successful_requests": cb.successful_requests,
            "llm_calls": cb.successful_requests,
            "tool_calls": tool_calls,
            "tools_used": tools_used
        }

        if enable_logging:
            try:
                logger = Logger(database_name="stock-assistant")
                logger.connect()
                logger.log_agent_run(user_query, response_content, metadata)
                logger.close()
            except Exception as e:
                print(f"⚠️  Warning: Failed to log to MotherDuck: {e}")

        return response_content, metadata



if __name__ == "__main__":
    pass
    # prompt = "Is MSFT still a buy"
    # response, cost = run_financial_agent(
    #     app,
    #     prompt
    # )

    # print("=" * 80)
    # print("RESPONSE:")
    # print("=" * 80)
    # print(response)
    # print("\n" + "=" * 80)
    # print("COST:")
    # print("=" * 80)
    # print(cost)
    # print(f"Cached companies: {get_cached_companies()}")
