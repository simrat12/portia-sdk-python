"""Uniswap trading module for the Portia SDK.

This module provides a wrapper around the Enso API for trading tokens on Uniswap V2.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Union, Dict, Any

import requests
from pydantic import BaseModel, Field

from portia.config import Config

# Set up logger
logger = logging.getLogger(__name__)


class UniswapRouteRequest(BaseModel):
    """Request model for Uniswap route calculation."""
    
    from_address: str = Field(..., alias="fromAddress", description="Ethereum address of the wallet to send the transaction from")
    amount_in: List[str] = Field(..., alias="amountIn", description="Amount of tokenIn to swap in wei")
    token_in: List[str] = Field(..., alias="tokenIn", description="Ethereum address of the token to swap from")
    token_out: List[str] = Field(..., alias="tokenOut", description="Ethereum address of the token to swap to")
    chain_id: int = Field(default=1, alias="chainId", description="Chain ID of the network to execute the transaction on")
    routing_strategy: Optional[str] = Field(None, alias="routingStrategy", description="Routing strategy to use")
    receiver: Optional[str] = Field(None, alias="receiver", description="Ethereum address of the receiver of the tokenOut")
    spender: Optional[str] = Field(None, alias="spender", description="Ethereum address of the spender of the tokenIn")
    min_amount_out: Optional[List[str]] = Field(None, alias="minAmountOut", description="Minimum amount out in wei")
    slippage: Optional[str] = Field(default="50", alias="slippage", description="Slippage in basis points (1/10000)")
    fee: Optional[List[str]] = Field(None, alias="fee", description="Fee in basis points (1/10000) for each amountIn value")
    fee_receiver: Optional[str] = Field(None, alias="feeReceiver", description="The Ethereum address that will receive the collected fee")
    ignore_aggregators: Optional[List[str]] = Field(None, alias="ignoreAggregators", description="A list of swap aggregators to be ignored from consideration")
    ignore_standards: Optional[List[str]] = Field(None, alias="ignoreStandards", description="A list of standards to be ignored from consideration")
    variable_estimates: Optional[Dict[str, Any]] = Field(
        None, 
        alias="variableEstimates", 
        description="An object or null, required by the Enso endpoint"
    )
    to_eoa: Optional[bool] = Field(
        None, 
        alias="toEoa", 
        description="Flag indicating if gained tokenOut is sent to EOA"
    )

    class Config:
        populate_by_name = True
        # ensures the JSON uses camelCase keys instead of snake_case


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
        logger.info("Initializing UniswapTrader")
        logger.debug(f"Config: {config}")
        logger.debug(f"Enso API URL: {enso_api_url}")
        
        self.config = config
        self.enso_api_key = enso_api_key
        self.enso_api_url = enso_api_url
        self.headers = {
            "Authorization": f"Bearer {enso_api_key}",
            "Content-Type": "application/json"
        }
        logger.debug("Headers set up with API key")
    
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
        ignore_standards: Optional[List[str]] = None,
        variable_estimates: Optional[Dict[str, Any]] = None,
        to_eoa: Optional[bool] = None
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
            variable_estimates: An object or null, required by the Enso endpoint
            to_eoa: Flag indicating if gained tokenOut is sent to EOA
            
        Returns:
            UniswapRouteResponse: The response from the Enso API
        """
        logger.info("Getting optimal route for Uniswap trade")
        logger.debug(f"from_address: {from_address}")
        logger.debug(f"amount_in: {amount_in}")
        logger.debug(f"token_in: {token_in}")
        logger.debug(f"token_out: {token_out}")
        logger.debug(f"chain_id: {chain_id}")
        logger.debug(f"routing_strategy: {routing_strategy}")
        logger.debug(f"receiver: {receiver}")
        logger.debug(f"spender: {spender}")
        logger.debug(f"min_amount_out: {min_amount_out}")
        logger.debug(f"slippage: {slippage}")
        logger.debug(f"fee: {fee}")
        logger.debug(f"fee_receiver: {fee_receiver}")
        logger.debug(f"ignore_aggregators: {ignore_aggregators}")
        logger.debug(f"ignore_standards: {ignore_standards}")
        logger.debug(f"variable_estimates: {variable_estimates}")
        logger.debug(f"to_eoa: {to_eoa}")
        
        # Convert single values to lists if needed
        if isinstance(amount_in, str):
            logger.debug(f"Converting amount_in from string to list: {amount_in}")
            amount_in = [amount_in]
        if isinstance(token_in, str):
            logger.debug(f"Converting token_in from string to list: {token_in}")
            token_in = [token_in]
        if isinstance(token_out, str):
            logger.debug(f"Converting token_out from string to list: {token_out}")
            token_out = [token_out]
        if isinstance(min_amount_out, str):
            logger.debug(f"Converting min_amount_out from string to list: {min_amount_out}")
            min_amount_out = [min_amount_out]
        if isinstance(fee, str):
            logger.debug(f"Converting fee from string to list: {fee}")
            fee = [fee]
        
        # Create the request
        logger.debug("Creating UniswapRouteRequest")
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
            ignore_standards=ignore_standards,
            variable_estimates=variable_estimates,
            to_eoa=to_eoa
        )
        
        # Convert request to dict for JSON serialization
        request_dict = request.model_dump(by_alias=True)
        logger.debug(f"Request dict: {request_dict}")
        
        # Send the request to the Enso API
        logger.info(f"Sending request to Enso API at {self.enso_api_url}/shortcuts/route")
        response = requests.post(
            f"{self.enso_api_url}/shortcuts/route",
            headers=self.headers,
            json=request_dict
        )
        
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Response text: {response.text}")
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Error from Enso API: {response.text}")
            raise Exception(f"Error from Enso API: {response.text}")
        
        # Parse the response
        logger.debug("Parsing response into UniswapRouteResponse")
        response_data = response.json()
        logger.debug(f"Response data: {response_data}")
        
        route_response = UniswapRouteResponse(**response_data)
        logger.info("Successfully created UniswapRouteResponse")
        logger.debug(f"Route response: {route_response}")
        
        return route_response
    
    def execute_trade(self, route_response: UniswapRouteResponse) -> str:
        """Execute a trade using the route response from the Enso API.
        
        Args:
            route_response: The route response from the Enso API
            
        Returns:
            str: The transaction hash
        """
        logger.info("Executing trade with route response")
        logger.debug(f"Route response: {route_response}")
        
        # This is a placeholder for the actual implementation
        # In a real implementation, you would use the tx object from the route response
        # to send a transaction to the blockchain
        
        # For now, we'll just return a dummy transaction hash
        dummy_tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        logger.info(f"Returning dummy transaction hash: {dummy_tx_hash}")
        return dummy_tx_hash 