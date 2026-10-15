import argparse
import flwr as fl
from strategy import TrustWeightedValidator


def get_strategy(strategy_name: str, min_clients: int, defense_mode: str = "soft"):
    if strategy_name == "fedavg":
        print("[SERVER CONFIG] Running Standard FedAvg (Undefended Control Baseline)...")
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
        )
    elif strategy_name == "fedmedian":
        print("[SERVER CONFIG] Running FedMedian (Geometric Byzantine Defense Baseline)...")
        return fl.server.strategy.FedMedian(
            fraction_fit=1.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
        )
    elif strategy_name == "fedrag":
        print(f"[SERVER CONFIG] Running FedRAG TrustWeightedValidator (Our Novelty) | defense_mode={defense_mode}...")
        return TrustWeightedValidator(
            fraction_fit=1.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            defense_mode=defense_mode,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedRAG Central Server")
    parser.add_argument("--strategy", type=str, default="fedrag", choices=["fedavg", "fedmedian", "fedrag"])
    parser.add_argument("--clients", type=int, default=3, help="Minimum clients required")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--defense-mode", type=str, default="soft", choices=["soft", "hard"],
                        help="FedRAG only: 'soft' = exponential trust decay, 'hard' = immediate permanent ban")
    args = parser.parse_args()

    print(f"Starting FedRAG Server with Strategy: {args.strategy.upper()} for {args.rounds} rounds...")

    strategy = get_strategy(args.strategy, args.clients, args.defense_mode)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )