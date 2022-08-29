# import sqlite3
# import os

# path = "backend/data/qanta.2018.04.18.sqlite3"

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from backend.database import Database, Question, User, Record


db = Database()
from backend.database import Question, User, Record


import sqlite3
from sqlite3 import Error

# db_url = "sqlite:///backend/data/database.db"
db_url = "backend/data/database.db"


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


# if __name__ == '__main__':
create_connection(db_url)
db.create_all()

print("created tables!")

# engine = create_engine('sqlite:///' + path, echo = True)
# meta = MetaData()


# users = Table(
#    'users', meta, 
#    Column('email', String, primary_key = True), 
#    Column('password', String), 
# )

# # class User(Base):
# #     __tablename__ = "users"
# #     email = Column(String, primary_key=True)
# #     password = Column(String)

# #     def __str__(self):
# #         return self.email


# meta.create_all(engine)