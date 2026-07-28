from pydantic import BaseModel, Field
from typing import Optional, List

class TransactionInput(BaseModel):
    transaction_amount: float = Field(..., ge=0.0, description="Amount of the transaction")
    merchant_category: str = Field(..., description="Category of merchant")
    merchant_country: str = Field(..., description="Country of merchant")
    payment_method: str = Field(..., description="Payment method used")
    device_type: str = Field(..., description="Type of device used")
    transaction_type: str = Field(..., description="Type of transaction (e.g. purchase, withdrawal, transfer)")
    customer_age: int = Field(..., ge=18, le=120, description="Age of the customer")
    account_age_days: int = Field(..., ge=0, description="Age of the account in days")
    previous_fraud_history: int = Field(..., ge=0, le=1, description="Previous fraud history (0 or 1)")
    transactions_last_24h: int = Field(..., ge=0, description="Number of transactions in last 24 hours")
    average_transaction_amount: float = Field(..., ge=0.0, description="Average transaction amount for customer")
    distance_from_home: float = Field(..., ge=0.0, description="Distance from customer home in km")
    card_present: int = Field(..., ge=0, le=1, description="Is card present physically (0 or 1)")
    international_transaction: int = Field(..., ge=0, le=1, description="Is transaction international (0 or 1)")

    # Optional fields that are automatically populated with defaults/heuristics if missing
    customer_id: Optional[str] = Field("cust_unknown", description="Unique identifier for customer")
    merchant_name: Optional[str] = Field(None, description="Name of the merchant")
    customer_country: Optional[str] = Field(None, description="Resident country of the customer")
    customer_gender: Optional[str] = Field("Other", description="Gender of customer (M, F, Other)")
    operating_system: Optional[str] = Field(None, description="Operating system of device")
    browser: Optional[str] = Field(None, description="Browser used for transaction")
    failed_transactions_last_24h: Optional[int] = Field(0, description="Number of failed attempts in last 24 hours")
    ip_risk_score: Optional[float] = Field(None, description="IP risk score (0.0 to 1.0)")
    device_trust_score: Optional[float] = Field(None, description="Device trust score (0.0 to 1.0)")
    merchant_risk_score: Optional[float] = Field(None, description="Merchant risk score (0.0 to 1.0)")

class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]