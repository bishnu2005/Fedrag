import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, EvaluateRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

VALID_DEFENSE_MODES = ("soft", "hard")

RESULTS_CSV_HEADER = [
    "Round", "Global_Recall_at_10", "Global_MRR_at_10", "Global_NDCG_at_10", "Defense_Mode", "Run_ID",
]
TRUST_CSV_HEADER = [
    "Run_ID", "Defense_Mode", "Round", "Client_ID", "Trust_Score", "Flagged_Malicious", "Accepted",
]


def append_csv_rows(csv_path: Path, header: List[str], rows: List[list]) -> None:
    """Append rows to csv_path, writing the header only when the file is first created.

    If the file already exists with a DIFFERENT header (e.g. the old 2-column
    federated_results.csv), it is moved aside to <name>_legacy[_N].csv instead of
    being appended to, because mixed column counts make pandas.read_csv fail.
    """
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != header:
            legacy_path = csv_path.with_name(f"{csv_path.stem}_legacy.csv")
            n = 1
            while legacy_path.exists():
                legacy_path = csv_path.with_name(f"{csv_path.stem}_legacy_{n}.csv")
                n += 1
            csv_path.rename(legacy_path)
            print(f"[LOGGER] {csv_path.name} had an older column layout; archived it as {legacy_path.name}")

    write_header = not csv_path.exists()
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


class TrustWeightedValidator(fl.server.strategy.FedAvg):
    def __init__(self, *args, defense_mode: str = "soft", **kwargs):
        if defense_mode not in VALID_DEFENSE_MODES:
            raise ValueError(f"defense_mode must be one of {VALID_DEFENSE_MODES}, got {defense_mode!r}")
        super().__init__(*args, **kwargs)
        self.defense_mode = defense_mode
        self.client_trust_scores = {}
        self.decay_factor = 0.5
        self.ban_threshold = 0.25

        # One id per server process, so repeated runs appended to the same CSVs stay separable.
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Build the notebooks directory if it doesn't exist yet
        NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        self.results_csv_path = NOTEBOOKS_DIR / "federated_results.csv"
        self.trust_csv_path = NOTEBOOKS_DIR / "trust_evolution.csv"

        print(f"[BYZANTINE DEFENDER] Defense mode: {self.defense_mode.upper()} | Run ID: {self.run_id}")

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        print(f"\n=======================================================")
        print(f"[BYZANTINE DEFENDER] Round {server_round}: Intercepted {len(results)} updates. "
              f"(mode={self.defense_mode})")
        print(f"=======================================================")

        for client_proxy, _ in results:
            if client_proxy.cid not in self.client_trust_scores:
                self.client_trust_scores[client_proxy.cid] = 1.0

        trusted_results = []
        trust_log_rows = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            current_trust = self.client_trust_scores[cid]

            # HARD mode: a banned client is excluded permanently, even if its update now looks clean.
            if self.defense_mode == "hard" and current_trust <= 0.0:
                print(f"  -> BANNED Hospital ID: {cid[-6:]} | Update discarded without inspection.")
                trust_log_rows.append([self.run_id, self.defense_mode, server_round, cid, 0.0, "", False])
                continue

            client_weights = parameters_to_ndarrays(fit_res.parameters)

            weight_spread = np.std(client_weights[0])
            is_malicious = bool(weight_spread > 0.10)
            accepted = False

            if is_malicious:
                print(f"  -> POISONED UPDATE DETECTED from Hospital ID: {cid[-6:]}")

                if self.defense_mode == "soft":
                    self.client_trust_scores[cid] = current_trust * self.decay_factor
                    new_trust = self.client_trust_scores[cid]
                    print(f"     DEGRADING TRUST: {current_trust} -> {new_trust}")

                    if new_trust <= self.ban_threshold:
                        print(f"     TRUST TOO LOW. Blocking this hospital's update this round.")
                    else:
                        print(f"      WARNING ISSUED. Data accepted this round, but trust severely damaged.")
                        trusted_results.append((client_proxy, fit_res))
                        accepted = True
                else:  # hard
                    self.client_trust_scores[cid] = 0.0
                    print(f"     HARD BAN: {current_trust} -> 0.0. Hospital permanently excluded.")
            else:
                print(f"  -> Clean Update from Hospital ID: {cid[-6:]} | Trust: {current_trust}")
                trusted_results.append((client_proxy, fit_res))
                accepted = True

            trust_log_rows.append([
                self.run_id, self.defense_mode, server_round, cid,
                self.client_trust_scores[cid], is_malicious, accepted,
            ])

        # Persist this round's trust state for the Trust Weight Evolution figure.
        append_csv_rows(self.trust_csv_path, TRUST_CSV_HEADER, trust_log_rows)
        print(f"[BYZANTINE DEFENDER] Trust scores for round {server_round} saved to {self.trust_csv_path}")

        if not trusted_results:
            print(
                "[BYZANTINE DEFENDER] ⚠️ No trusted updates this round! Forcing fallback to include highest-trust client to prevent deadlock.")
            # Fallback: Sort results by trust score (if available) or just take the first client's update
            if results:
                # If your results object contains client metadata, grab the first one as a safety net
                trusted_results = [results[0]]
            else:
                return None, {}

        print(f"\n[BYZANTINE DEFENDER] Proceeding with Safe Aggregation over "
              f"{len(trusted_results)}/{len(results)} clients...")
        return super().aggregate_fit(server_round, trusted_results, failures)

    # YOUR NEW CSV GENERATOR: Intercepting the Evaluation Scores
    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated_loss, aggregated_metrics

        # Unweighted mean over every client that returned metrics (same convention as the
        # original Recall@10 logging). Attack clients only report recall@10=0.0, so their
        # missing mrr / ndcg@10 also default to 0.0 and the three columns stay comparable.
        def mean_metric(key: str) -> float:
            values = [float(evaluate_res.metrics.get(key, 0.0)) for _, evaluate_res in results]
            return sum(values) / len(values) if values else 0.0

        avg_recall = mean_metric("recall@10")
        avg_mrr = mean_metric("mrr")
        avg_ndcg = mean_metric("ndcg@10")

        print(f"\n=======================================================")
        print(f"[SERVER METRICS] Round {server_round} | Recall@10: {avg_recall:.4f} | "
              f"MRR@10: {avg_mrr:.4f} | NDCG@10: {avg_ndcg:.4f}")
        print(f"=======================================================")

        # Automatically write the scores to your research CSV
        append_csv_rows(
            self.results_csv_path,
            RESULTS_CSV_HEADER,
            [[server_round, avg_recall, avg_mrr, avg_ndcg, self.defense_mode, self.run_id]],
        )
        print(f"Metrics successfully saved to {self.results_csv_path}")

        return aggregated_loss, {"recall@10": avg_recall, "mrr": avg_mrr, "ndcg@10": avg_ndcg}