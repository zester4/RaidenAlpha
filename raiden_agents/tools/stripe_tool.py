import stripe
from typing import Dict, Any
from .base_tool import Tool, ToolExecutionError

class StripePaymentTool(Tool):
    def __init__(self, api_key: str):
        super().__init__(
            name="stripe",
            description="Handle Stripe payment operations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["CREATE_PAYMENT", "REFUND", "GET_PAYMENT", "CREATE_CUSTOMER", "GET_CUSTOMER"]
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount in cents",
                        "optional": True
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency code",
                        "optional": True
                    },
                    "payment_id": {
                        "type": "string",
                        "description": "Stripe payment intent ID",
                        "optional": True
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "Stripe customer ID",
                        "optional": True
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer email",
                        "optional": True
                    }
                },
                "required": ["operation"]
            }
        )
        stripe.api_key = api_key

    def execute(self, **kwargs) -> str:
        self.validate_args(kwargs)
        operation = kwargs["operation"]

        try:
            if operation == "CREATE_PAYMENT":
                return self._create_payment(
                    amount=kwargs.get("amount"),
                    currency=kwargs.get("currency", "usd")
                )

            elif operation == "REFUND":
                return self._refund_payment(kwargs.get("payment_id"))

            elif operation == "GET_PAYMENT":
                return self._get_payment(kwargs.get("payment_id"))

            elif operation == "CREATE_CUSTOMER":
                return self._create_customer(kwargs.get("email"))

            elif operation == "GET_CUSTOMER":
                return self._get_customer(kwargs.get("customer_id"))

            else:
                raise ToolExecutionError(f"Unsupported operation: {operation}")

        except stripe.error.StripeError as e:
            raise ToolExecutionError(f"Stripe operation failed: {str(e)}")

    def _create_payment(self, amount: int, currency: str) -> str:
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=currency
        )
        return f"Payment intent created: {intent.id}"

    def _refund_payment(self, payment_id: str) -> str:
        refund = stripe.Refund.create(payment_intent=payment_id)
        return f"Refund created: {refund.id}"

    def _get_payment(self, payment_id: str) -> str:
        intent = stripe.PaymentIntent.retrieve(payment_id)
        return f"Payment status: {intent.status}"

    def _create_customer(self, email: str) -> str:
        customer = stripe.Customer.create(email=email)
        return f"Customer created: {customer.id}"

    def _get_customer(self, customer_id: str) -> str:
        customer = stripe.Customer.retrieve(customer_id)
        return f"Customer: {customer.email}"