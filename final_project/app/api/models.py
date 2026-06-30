"""SQLAlchemy ORM model."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.api.db import Base


class Generation(Base):
    __tablename__ = "generations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    model_version = Column(String(128), nullable=False)
    latency_ms = Column(Integer)
    status = Column(String(32), default="success")
