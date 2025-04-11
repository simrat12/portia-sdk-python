#!/usr/bin/env python
"""Example script demonstrating how to use the UniswapTrader class."""

from portia import Config, UniswapTrader

def main():
    """Run the example."""
    # Create a configuration
    config = Config.from_default(
        llm_provider="OPENAI",
        openai_api_key="your-openai-api-key"
    )
    
    # Create a UniswapTrader instance
    trader = UniswapTrader(
        config=config,
        enso_api_key="your-enso-api-key"
    )
    
    # Define the trading parameters
    from_address = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"  # Example address
    amount_in = "1000000000000000000"  # 1 ETH in wei
    token_in = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"  # ETH
    token_out = "0x6b175474e89094c44da98b954eedeac495271d0f"  # DAI
    
    # Get the optimal route
    route_response = trader.get_optimal_route(
        from_address=from_address,
        amount_in=amount_in,
        token_in=token_in,
        token_out=token_out,
        slippage="50"  # 0.5% slippage
    )
    
    # Print the route details
    print(f"Gas: {route_response.gas}")
    print(f"Amount out: {route_response.amount_out}")
    print(f"Price impact: {route_response.price_impact}%")
    print(f"Fee amount: {route_response.fee_amount}")
    
    # Execute the trade (in a real implementation, this would send a transaction to the blockchain)
    tx_hash = trader.execute_trade(route_response)
    print(f"Transaction hash: {tx_hash}")

if __name__ == "__main__":
    main() 