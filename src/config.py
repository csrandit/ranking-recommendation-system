import os


class Settings:
    DATA_PATH = os.getenv("DATA_PATH", "data/raw/ml-100k")
    K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", 40))


settings = Settings()