import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CODE'))

from fastapi import FastAPI
from CODE.api.router.stocks import router as stocks_router

app = FastAPI()

app.include_router(stocks_router)

# Pour lancer l'application, utilisez uvicorn: uvicorn api.main:app --reload
