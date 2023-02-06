import schema
from sqlalchemy import create_engine

if __name__ == "__main__":
    engine = create_engine("sqlite:///db/data.db", echo=True)
    # schema.Base.metadata.drop_all(engine)
    schema.Base.metadata.create_all(engine)