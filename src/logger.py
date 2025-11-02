import logging 
import os
from datetime import datetime

folder = "logs"
os.makedirs(folder, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(folder, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    logging.info("Logging has started")