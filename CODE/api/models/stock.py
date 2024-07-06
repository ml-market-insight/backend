from pydantic import BaseModel
from typing import List

class StockSelection(BaseModel):
    tickers: List[str]
