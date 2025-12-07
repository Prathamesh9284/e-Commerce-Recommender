from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import os
import logging
from routers import recommend, product, behavior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)
app.include_router(product.router)
app.include_router(behavior.router)

