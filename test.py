from backend.agent import create_financial_agent
from backend.utils import create_state_graph

if __name__ == "__main__":
    app = create_financial_agent()
    create_state_graph(app)

    # # Test queries that will trigger period errors
    # test_queries = [
    #     "Show me JBLU stock performance for the past week",  # Will use '1w'
    #     # "Get AAPL data for the last 2 weeks",  # Will use '2w'
    #     # "What's TSLA stock doing this month?",  # Might use 'month'
    # ]

    # for i, query in enumerate(test_queries, 1):
    #     # print(f"\n{'='*80}")
    #     # print(f"Test {i}: {query}")
    #     # print('='*80)

    #     response, metadata = run_financial_agent(app, query, enable_logging=False)

    #     # print("\nRESPONSE:")
    #     # print(response)

    #     # print("\nMETADATA:")
    #     # print(f"  Errors corrected: {metadata.get('errors_corrected', 0)}")
    #     # print(f"  Parameter corrections: {metadata.get('parameter_corrections', {})}")
    #     # print(f"  Total tokens: {metadata['total_tokens']}")
    #     # print(f"  Tool calls: {metadata['tool_calls']}")
    #     print(f"  Tools used: {metadata['tools_used']}")
