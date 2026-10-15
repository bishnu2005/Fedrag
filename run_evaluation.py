import subprocess
import time
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXEC = sys.executable  # Uses current active virtualenv Python


def run_single_experiment(strategy: str, attack_script: str, run_id: int, defense_mode: str = "soft",
                          noise_skew: float = 0.0):
    print(f"\n{'=' * 70}")
    print(
        f"▶️ RUNNING TRIAL {run_id} | Strategy: {strategy.upper()} ({defense_mode}) | Attack: {attack_script} | Skew: {noise_skew}")
    print(f"{'=' * 70}")

    # 1. Start Central Server
    server_cmd = [
        PYTHON_EXEC, str(PROJECT_ROOT / "fedrag" / "server" / "flower_server.py"),
        "--strategy", strategy,
        "--clients", "3",
        "--rounds", "3",
        "--defense-mode", defense_mode,
    ]
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(5)  # Increased to 5 seconds to ensure server socket opens

    # 2. Start Honest Hospital 1 (TREC-COVID - Clean Baseline)
    client1_cmd = [
        PYTHON_EXEC, str(PROJECT_ROOT / "fedrag" / "client" / "flower_client.py"),
        "--client-id", "Hospital_A_Clean",
        "--dataset", "trec-covid"
    ]
    c1_proc = subprocess.Popen(client1_cmd)

    # 3. Start Honest Hospital 2 (NFCorpus - Testing False Positive Rate)
    client2_cmd = [
        PYTHON_EXEC, str(PROJECT_ROOT / "fedrag" / "client" / "flower_client.py"),
        "--client-id", "Hospital_B_Skewed",
        "--dataset", "nfcorpus",
        "--noise-skew", str(noise_skew)  # New argument added here
    ]
    c2_proc = subprocess.Popen(client2_cmd)

    # 4. Start Attacker (Hospital 3 - SciFact)
    attacker_cmd = [
        PYTHON_EXEC, str(PROJECT_ROOT / "fedrag" / "attacks" / attack_script),
        "--dataset", "scifact"  # Passed in case the attack script requires it
    ]
    c3_proc = subprocess.Popen(attacker_cmd)

    # Wait for all processes to finish
    server_proc.wait()
    c1_proc.wait()
    c2_proc.wait()
    c3_proc.wait()

    print(f"Trial {run_id} complete.")


def main():
    trials = [
        # ==========================================
        # Phase 1: Baselines (No Defense)
        # ==========================================
        ("fedavg", "sign_flip.py", "soft", 0.0),
        ("fedavg", "gaussian_noise.py", "soft", 0.0),

        # ==========================================
        # Phase 2: Standard Geometric Defenses
        # ==========================================
        ("fedmedian", "sign_flip.py", "soft", 0.0),
        ("fedmedian", "gaussian_noise.py", "soft", 0.0),

        # ==========================================
        # Phase 3: FedRAG Core (Soft Decay vs Standard Attacks)
        # ==========================================
        ("fedrag", "sign_flip.py", "soft", 0.0),
        ("fedrag", "gaussian_noise.py", "soft", 0.0),

        # ==========================================
        # Phase 4: The Krum-Bypass Proof (Semantic Scrambling)
        # ==========================================
        ("fedmedian", "krum_bypass.py", "soft", 0.0),
        ("fedrag", "krum_bypass.py", "soft", 0.0),

        # ==========================================
        # Phase 5: False Positive Rate Test (40% Honest Client Skew)
        # ==========================================
        ("fedmedian", "sign_flip.py", "soft", 0.4),
        ("fedrag", "sign_flip.py", "soft", 0.4),

        # ==========================================
        # Phase 6: Hard Ban Ablation (Rigid Ban vs Soft Decay)
        # ==========================================
        ("fedrag", "sign_flip.py", "hard", 0.0),
        ("fedrag", "gaussian_noise.py", "hard", 0.0),
        ("fedrag", "krum_bypass.py", "hard", 0.0),
    ]

    NUM_REPETITIONS = 1  # Keep at 1 until you verify the logs look correct

    for rep in range(1, NUM_REPETITIONS + 1):
        for strategy, attack, defense_mode, noise_skew in trials:
            run_single_experiment(strategy, attack, run_id=rep, defense_mode=defense_mode, noise_skew=noise_skew)
            time.sleep(5)  # Cooldown between runs

    print("\nALL EXPERIMENTAL BENCHMARKS COMPLETED!")


if __name__ == "__main__":
    main()