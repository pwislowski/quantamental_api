from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.db import get_db


class Service:
    def __init__(self, db: Annotated[Session, Depends(get_db, use_cache=True)]):
        self.db = db
