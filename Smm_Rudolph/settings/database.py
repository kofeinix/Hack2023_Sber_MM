from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.types import BLOB
from sqlalchemy.orm import relationship
import uuid

SQLALCHEMY_DATABASE_URL = "sqlite:///./db.sqlite3"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = Session()

Base = declarative_base()


class Pages(Base):
    __tablename__ = "SMMparser_pages"
    id = Column(String(36), primary_key=True, default=str(uuid.uuid4()), nullable=False)
    categ_root = Column(String(100), nullable=True)
    categ1 = Column(String(100), nullable=True)
    categ2 = Column(String(100), nullable=True)
    categ3 = Column(String(100), nullable=True)
    categ4 = Column(String(100), nullable=True)
    name = Column(String(100), unique=True)
    url = Column(String(100), nullable=True)
    properties = Column(Text, nullable=True)
    jsfiled = Column(Text, nullable=True)
class Images(Base):
    __tablename__ = "SMMparser_images"
    id = Column(String(36), primary_key=True, default=str(uuid.uuid4()), nullable=False)
    url = Column(String(500), unique=True)
    page_id = Column(String(36), ForeignKey('SMMparser_pages.id'))
    page = relationship("Pages", backref="images")
    imageio = Column(BLOB, nullable=True)