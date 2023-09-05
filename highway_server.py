from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Parameters
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import GetParametersIns
from flwr.server.utils.tensorboard import tensorboard
from flwr.server.strategy import FedAvg, FedAdam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="modified model name", type=str, nargs='?')
args = parser.parse_args()

def main():
    total_rounds = 40
    # Decorated strategy
    strategy = FedAvg(min_fit_clients=2,
                      min_evaluate_clients=2,
                      min_available_clients=2,
                      )

               
    # Start Flower server
    fl.server.start_server(
        server_address="192.168.1.85:8080",
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
    )
    
if __name__ == "__main__":
    main()
