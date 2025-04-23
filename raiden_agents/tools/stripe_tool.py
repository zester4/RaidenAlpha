import stripe
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

class StripePaymentTool(Tool):
    """Enhanced Stripe Tool with comprehensive payment and subscription features"""
    
    def __init__(self, api_key: str, webhook_secret: Optional[str] = None):
        super().__init__(
            name="stripe",
            description="Comprehensive Stripe payment processing and subscription management",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            # Payment Operations
                            "CREATE_PAYMENT", "CONFIRM_PAYMENT", "CANCEL_PAYMENT", "REFUND_PAYMENT",
                            "GET_PAYMENT", "LIST_PAYMENTS",
                            # Customer Operations
                            "CREATE_CUSTOMER", "UPDATE_CUSTOMER", "DELETE_CUSTOMER", "GET_CUSTOMER",
                            "LIST_CUSTOMERS", "ADD_PAYMENT_METHOD", "REMOVE_PAYMENT_METHOD",
                            # Subscription Operations
                            "CREATE_SUBSCRIPTION", "UPDATE_SUBSCRIPTION", "CANCEL_SUBSCRIPTION",
                            "GET_SUBSCRIPTION", "LIST_SUBSCRIPTIONS", "PAUSE_SUBSCRIPTION",
                            "RESUME_SUBSCRIPTION",
                            # Product & Price Operations
                            "CREATE_PRODUCT", "UPDATE_PRODUCT", "DELETE_PRODUCT", "GET_PRODUCT",
                            "LIST_PRODUCTS", "CREATE_PRICE", "UPDATE_PRICE", "GET_PRICE",
                            "LIST_PRICES",
                            # Invoice Operations
                            "CREATE_INVOICE", "PAY_INVOICE", "VOID_INVOICE", "GET_INVOICE",
                            "LIST_INVOICES",
                            # Dispute Operations
                            "UPDATE_DISPUTE", "CLOSE_DISPUTE", "GET_DISPUTE", "LIST_DISPUTES",
                            # Reporting Operations
                            "GET_BALANCE", "GET_TRANSACTION", "LIST_TRANSACTIONS",
                            # Webhook Operations
                            "VERIFY_WEBHOOK", "PROCESS_WEBHOOK"
                        ]
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount in smallest currency unit (e.g., cents)",
                        "optional": True
                    },
                    "currency": {
                        "type": "string",
                        "description": "Three-letter ISO currency code",
                        "optional": True
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "Stripe customer ID",
                        "optional": True
                    },
                    "payment_method": {
                        "type": "string",
                        "description": "Payment method ID or type",
                        "optional": True
                    },
                    "payment_intent_id": {
                        "type": "string",
                        "description": "Payment intent ID",
                        "optional": True
                    },
                    "subscription_id": {
                        "type": "string",
                        "description": "Subscription ID",
                        "optional": True
                    },
                    "product_id": {
                        "type": "string",
                        "description": "Product ID",
                        "optional": True
                    },
                    "price_id": {
                        "type": "string",
                        "description": "Price ID",
                        "optional": True
                    },
                    "invoice_id": {
                        "type": "string",
                        "description": "Invoice ID",
                        "optional": True
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata for Stripe objects",
                        "optional": True
                    },
                    "webhook_payload": {
                        "type": "object",
                        "description": "Webhook event payload",
                        "optional": True
                    },
                    "webhook_signature": {
                        "type": "string",
                        "description": "Webhook signature header",
                        "optional": True
                    }
                },
                "required": ["operation"]
            }
        )
        stripe.api_key = api_key
        self.webhook_secret = webhook_secret

    def execute(self, **kwargs) -> Union[str, Dict[str, Any]]:
        """Execute Stripe operations based on provided parameters"""
        self.validate_args(kwargs)
        operation = kwargs.pop("operation")

        try:
            # Payment Operations
            if operation == "CREATE_PAYMENT":
                return self._create_payment(**kwargs)
            elif operation == "CONFIRM_PAYMENT":
                return self._confirm_payment(**kwargs)
            elif operation == "CANCEL_PAYMENT":
                return self._cancel_payment(**kwargs)
            elif operation == "REFUND_PAYMENT":
                return self._refund_payment(**kwargs)
            elif operation == "GET_PAYMENT":
                return self._get_payment(**kwargs)
            elif operation == "LIST_PAYMENTS":
                return self._list_payments(**kwargs)

            # Customer Operations
            elif operation == "CREATE_CUSTOMER":
                return self._create_customer(**kwargs)
            elif operation == "UPDATE_CUSTOMER":
                return self._update_customer(**kwargs)
            elif operation == "DELETE_CUSTOMER":
                return self._delete_customer(**kwargs)
            elif operation == "GET_CUSTOMER":
                return self._get_customer(**kwargs)
            elif operation == "LIST_CUSTOMERS":
                return self._list_customers(**kwargs)
            elif operation == "ADD_PAYMENT_METHOD":
                return self._add_payment_method(**kwargs)
            elif operation == "REMOVE_PAYMENT_METHOD":
                return self._remove_payment_method(**kwargs)

            # Subscription Operations
            elif operation == "CREATE_SUBSCRIPTION":
                return self._create_subscription(**kwargs)
            elif operation == "UPDATE_SUBSCRIPTION":
                return self._update_subscription(**kwargs)
            elif operation == "CANCEL_SUBSCRIPTION":
                return self._cancel_subscription(**kwargs)
            elif operation == "GET_SUBSCRIPTION":
                return self._get_subscription(**kwargs)
            elif operation == "LIST_SUBSCRIPTIONS":
                return self._list_subscriptions(**kwargs)
            elif operation == "PAUSE_SUBSCRIPTION":
                return self._pause_subscription(**kwargs)
            elif operation == "RESUME_SUBSCRIPTION":
                return self._resume_subscription(**kwargs)

            # Product & Price Operations
            elif operation == "CREATE_PRODUCT":
                return self._create_product(**kwargs)
            elif operation == "UPDATE_PRODUCT":
                return self._update_product(**kwargs)
            elif operation == "DELETE_PRODUCT":
                return self._delete_product(**kwargs)
            elif operation == "GET_PRODUCT":
                return self._get_product(**kwargs)
            elif operation == "LIST_PRODUCTS":
                return self._list_products(**kwargs)
            elif operation == "CREATE_PRICE":
                return self._create_price(**kwargs)
            elif operation == "UPDATE_PRICE":
                return self._update_price(**kwargs)
            elif operation == "GET_PRICE":
                return self._get_price(**kwargs)
            elif operation == "LIST_PRICES":
                return self._list_prices(**kwargs)

            # Invoice Operations
            elif operation == "CREATE_INVOICE":
                return self._create_invoice(**kwargs)
            elif operation == "PAY_INVOICE":
                return self._pay_invoice(**kwargs)
            elif operation == "VOID_INVOICE":
                return self._void_invoice(**kwargs)
            elif operation == "GET_INVOICE":
                return self._get_invoice(**kwargs)
            elif operation == "LIST_INVOICES":
                return self._list_invoices(**kwargs)

            # Dispute Operations
            elif operation == "UPDATE_DISPUTE":
                return self._update_dispute(**kwargs)
            elif operation == "CLOSE_DISPUTE":
                return self._close_dispute(**kwargs)
            elif operation == "GET_DISPUTE":
                return self._get_dispute(**kwargs)
            elif operation == "LIST_DISPUTES":
                return self._list_disputes(**kwargs)

            # Reporting Operations
            elif operation == "GET_BALANCE":
                return self._get_balance()
            elif operation == "GET_TRANSACTION":
                return self._get_transaction(**kwargs)
            elif operation == "LIST_TRANSACTIONS":
                return self._list_transactions(**kwargs)

            # Webhook Operations
            elif operation == "VERIFY_WEBHOOK":
                return self._verify_webhook(**kwargs)
            elif operation == "PROCESS_WEBHOOK":
                return self._process_webhook(**kwargs)

            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")

        except stripe.error.StripeError as e:
            raise ToolExecutionError(f"Stripe operation failed: {str(e)}")

    # Payment Methods
    def _create_payment(self, amount: int, currency: str, 
                       customer_id: Optional[str] = None,
                       payment_method: Optional[str] = None,
                       metadata: Optional[Dict] = None,
                       **kwargs) -> Dict[str, Any]:
        """Create a payment intent"""
        params = {
            "amount": amount,
            "currency": currency,
            "automatic_payment_methods": {"enabled": True}
        }
        if customer_id:
            params["customer"] = customer_id
        if payment_method:
            params["payment_method"] = payment_method
        if metadata:
            params["metadata"] = metadata

        return stripe.PaymentIntent.create(**params)

    def _confirm_payment(self, payment_intent_id: str, 
                        payment_method: Optional[str] = None, 
                        **kwargs) -> Dict[str, Any]:
        """Confirm a payment intent"""
        return stripe.PaymentIntent.confirm(
            payment_intent_id,
            payment_method=payment_method
        )

    def _cancel_payment(self, payment_intent_id: str, 
                       cancellation_reason: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """Cancel a payment intent"""
        return stripe.PaymentIntent.cancel(
            payment_intent_id,
            cancellation_reason=cancellation_reason
        )

    def _refund_payment(self, payment_intent_id: str, 
                       amount: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """Refund a payment"""
        params = {"payment_intent": payment_intent_id}
        if amount:
            params["amount"] = amount
        return stripe.Refund.create(**params)

    # Customer Methods
    def _create_customer(self, email: str, name: Optional[str] = None,
                        payment_method: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        **kwargs) -> Dict[str, Any]:
        """Create a new customer"""
        params = {"email": email}
        if name:
            params["name"] = name
        if payment_method:
            params["payment_method"] = payment_method
        if metadata:
            params["metadata"] = metadata
        return stripe.Customer.create(**params)

    def _update_customer(self, customer_id: str,
                        email: Optional[str] = None,
                        name: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        **kwargs) -> Dict[str, Any]:
        """Update customer details"""
        params = {}
        if email:
            params["email"] = email
        if name:
            params["name"] = name
        if metadata:
            params["metadata"] = metadata
        return stripe.Customer.modify(customer_id, **params)

    # Subscription Methods
    def _create_subscription(self, customer_id: str, price_id: str,
                           trial_period_days: Optional[int] = None,
                           metadata: Optional[Dict] = None,
                           **kwargs) -> Dict[str, Any]:
        """Create a new subscription"""
        params = {
            "customer": customer_id,
            "items": [{"price": price_id}]
        }
        if trial_period_days:
            params["trial_period_days"] = trial_period_days
        if metadata:
            params["metadata"] = metadata
        return stripe.Subscription.create(**params)

    def _pause_subscription(self, subscription_id: str,
                          **kwargs) -> Dict[str, Any]:
        """Pause a subscription"""
        return stripe.Subscription.modify(
            subscription_id,
            pause_collection={"behavior": "void"}
        )

    # Product Methods
    def _create_product(self, name: str, description: Optional[str] = None,
                       metadata: Optional[Dict] = None,
                       **kwargs) -> Dict[str, Any]:
        """Create a new product"""
        params = {"name": name}
        if description:
            params["description"] = description
        if metadata:
            params["metadata"] = metadata
        return stripe.Product.create(**params)

    # Price Methods
    def _create_price(self, product_id: str, unit_amount: int,
                     currency: str, recurring: Optional[Dict] = None,
                     metadata: Optional[Dict] = None,
                     **kwargs) -> Dict[str, Any]:
        """Create a new price"""
        params = {
            "product": product_id,
            "unit_amount": unit_amount,
            "currency": currency
        }
        if recurring:
            params["recurring"] = recurring
        if metadata:
            params["metadata"] = metadata
        return stripe.Price.create(**params)

    # Invoice Methods
    def _create_invoice(self, customer_id: str,
                       description: Optional[str] = None,
                       metadata: Optional[Dict] = None,
                       **kwargs) -> Dict[str, Any]:
        """Create a new invoice"""
        params = {"customer": customer_id}
        if description:
            params["description"] = description
        if metadata:
            params["metadata"] = metadata
        return stripe.Invoice.create(**params)

    # Webhook Methods
    def _verify_webhook(self, payload: str, sig_header: str,
                       **kwargs) -> bool:
        """Verify webhook signature"""
        try:
            stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            return True
        except Exception:
            return False

    # Utility Methods
    def _format_currency(self, amount: int, currency: str) -> str:
        """Format currency amount"""
        currencies = {
            'usd': '$', 'eur': '€', 'gbp': '£',
            'jpy': '¥', 'aud': 'A$', 'cad': 'C$'
        }
        symbol = currencies.get(currency.lower(), '')
        amount_float = amount / 100.0
        return f"{symbol}{amount_float:.2f}"

    def _handle_error(self, error: stripe.error.StripeError) -> Dict[str, Any]:
        """Handle Stripe errors"""
        error_types = {
            stripe.error.CardError: "Card Error",
            stripe.error.InvalidRequestError: "Invalid Request",
            stripe.error.AuthenticationError: "Authentication Error",
            stripe.error.APIConnectionError: "API Connection Error",
            stripe.error.StripeError: "Generic Stripe Error"
        }
        error_type = error_types.get(type(error), "Unknown Error")
        return {
            "error": True,
            "type": error_type,
            "message": str(error),
            "code": error.code if hasattr(error, 'code') else None,
            "param": error.param if hasattr(error, 'param') else None,
            "timestamp": datetime.now().isoformat()
        }