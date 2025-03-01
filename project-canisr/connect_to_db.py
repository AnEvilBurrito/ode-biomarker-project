from sqlalchemy import create_engine

# load .env file to get the current 

sync_engine = create_engine(
    "postgresql+pg8000://canisr:canisr@192.168.3.106:9080/db"
)

with sync_engine.connect() as conn:
    print('Connection Successful')
