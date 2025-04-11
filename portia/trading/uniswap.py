"""Uniswap trading module for the Portia SDK.

This module provides a wrapper around the Enso API for trading tokens on Uniswap V2.
"""

from __future__ import annotations

import json
from typing import List, Optional, Union

import requests
from pydantic import BaseModel, Field

from portia.config import Config


class UniswapRouteRequest(BaseModel):
    """Request model for Uniswap route calculation."""
    
    from_address: str = Field(..., description="Ethereum address of the wallet to send the transaction from")
    amount_in: List[str] = Field(..., description="Amount of tokenIn to swap in wei")
    token_in: List[str] = Field(..., description="Ethereum address of the token to swap from")
    token_out: List[str] = Field(..., description="Ethereum address of the token to swap to")
    chain_id: int = Field(default=1, description="Chain ID of the network to execute the transaction on")
    routing_strategy: Optional[str] = Field(None, description="Routing strategy to use")
    receiver: Optional[str] = Field(None, description="Ethereum address of the receiver of the tokenOut")
    spender: Optional[str] = Field(None, description="Ethereum address of the spender of the tokenIn")
    min_amount_out: Optional[List[str]] = Field(None, description="Minimum amount out in wei")
    slippage: Optional[str] = Field(default="50", description="Slippage in basis points (1/10000)")
    fee: Optional[List[str]] = Field(None, description="Fee in basis points (1/10000) for each amountIn value")
    fee_receiver: Optional[str] = Field(None, description="The Ethereum address that will receive the collected fee")
    ignore_aggregators: Optional[List[str]] = Field(None, description="A list of swap aggregators to be ignored from consideration")
    ignore_standards: Optional[List[str]] = Field(None, description="A list of standards to be ignored from consideration")


class UniswapRouteResponse(BaseModel):
    """Response model for Uniswap route calculation."""
    
    gas: str
    amount_out: dict
    price_impact: float
    fee_amount: List[str]
    created_at: int
    tx: dict
    route: List[dict]


class UniswapTrader:
    """A wrapper around the Enso API for trading tokens on Uniswap V2."""
    
    def __init__(self, config: Config, enso_api_key: str, enso_api_url: str = "https://api.enso.finance/api/v1"):
        """Initialize the UniswapTrader.
        
        Args:
            config: The Portia SDK configuration
            enso_api_key: The Enso API key
            enso_api_url: The Enso API URL
        """
        self.config = config
        self.enso_api_key = enso_api_key
        self.enso_api_url = enso_api_url
        self.headers = {
            "Authorization": f"Bearer {enso_api_key}",
            "Content-Type": "application/json"
        }
    
    def get_optimal_route(
        self,
        from_address: str,
        amount_in: Union[str, List[str]],
        token_in: Union[str, List[str]],
        token_out: Union[str, List[str]],
        chain_id: int = 1,
        routing_strategy: Optional[str] = None,
        receiver: Optional[str] = None,
        spender: Optional[str] = None,
        min_amount_out: Optional[Union[str, List[str]]] = None,
        slippage: Optional[str] = "50",
        fee: Optional[Union[str, List[str]]] = None,
        fee_receiver: Optional[str] = None,
        ignore_aggregators: Optional[List[str]] = None,
        ignore_standards: Optional[List[str]] = None
    ) -> UniswapRouteResponse:
        """Get the optimal route for trading tokens on Uniswap V2.
        
        Args:
            from_address: Ethereum address of the wallet to send the transaction from
            amount_in: Amount of tokenIn to swap in wei
            token_in: Ethereum address of the token to swap from
            token_out: Ethereum address of the token to swap to
            chain_id: Chain ID of the network to execute the transaction on
            routing_strategy: Routing strategy to use
            receiver: Ethereum address of the receiver of the tokenOut
            spender: Ethereum address of the spender of the tokenIn
            min_amount_out: Minimum amount out in wei
            slippage: Slippage in basis points (1/10000)
            fee: Fee in basis points (1/10000) for each amountIn value
            fee_receiver: The Ethereum address that will receive the collected fee
            ignore_aggregators: A list of swap aggregators to be ignored from consideration
            ignore_standards: A list of standards to be ignored from consideration
            
        Returns:
            UniswapRouteResponse: The response from the Enso API
        """
        # Convert single values to lists if needed
        if isinstance(amount_in, str):
            amount_in = [amount_in]
        if isinstance(token_in, str):
            token_in = [token_in]
        if isinstance(token_out, str):
            token_out = [token_out]
        if isinstance(min_amount_out, str):
            min_amount_out = [min_amount_out]
        if isinstance(fee, str):
            fee = [fee]
        
        # Create the request
        request = UniswapRouteRequest(
            from_address=from_address,
            amount_in=amount_in,
            token_in=token_in,
            token_out=token_out,
            chain_id=chain_id,
            routing_strategy=routing_strategy,
            receiver=receiver,
            spender=spender,
            min_amount_out=min_amount_out,
            slippage=slippage,
            fee=fee,
            fee_receiver=fee_receiver,
            ignore_aggregators=ignore_aggregators,
            ignore_standards=ignore_standards
        )
        
        # Send the request to the Enso API
        response = requests.post(
            f"{self.enso_api_url}/shortcuts/route",
            headers=self.headers,
            data=request.model_dump_json()
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Error from Enso API: {response.text}")
        
        # Parse the response
        return UniswapRouteResponse(**response.json())
    
    def execute_trade(self, route_response: UniswapRouteResponse) -> str:
        """Execute a trade using the route response from the Enso API.
        
        Args:
            route_response: The route response from the Enso API
            
        Returns:
            str: The transaction hash
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, you would use the tx object from the route response
        # to send a transaction to the blockchain
        
        # For now, we'll just return a dummy transaction hash
        return "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef" 