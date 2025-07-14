import subprocess
import json
import sys
import time
import os
import socket

def wait_for_port_release(port, host='127.0.0.1', timeout=10):
    """Wait until the given port is free (or until timeout)."""
    start_time = time.time()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                # If bind succeeds, port is free.
                break
            except socket.error:
                pass  # Port is still in use.
        if time.time() - start_time > timeout:
            print(f"Port {port} is still in use after {timeout} seconds.")
            raise ValueError("Timeout")
        time.sleep(1)
    print(f"Port {port} is now free.")

def run_instance(instance_config):
    """Runs one complete instance (Phase 1 and Phase 2) sequentially based on instance_config."""
    instance_id = instance_config.get("instance_id", "instance")
    server_cfg = instance_config['server']
    clients_cfg = instance_config['clients']
    nofed_cfg = instance_config['nofed_client']

    print(f"[{instance_id}] Starting Phase 1 (weighted mode, no folder)...")
    # ---------------- Phase 1: Launch server in weighted mode (w) without folder ----------------
    server_args_phase1 = [
        "python3", "server.py",
        "--p", str(server_cfg['port']),
        "--c", str(server_cfg['num_clients']),
        "--i", str(server_cfg.get('num_iterations', 100)),
        "--f", str(server_cfg.get('force_rounds', 15)),
        "--m", server_cfg.get('model_type', 'SGDC'),
        "--w", server_cfg.get('first_phase_mode', 'w'),
        "--combs", str(server_cfg.get('combinations', 1)),
        "--lr", str(server_cfg.get('lr', 0.10))
    ]
    print(f"[{instance_id}] Phase 1 server args: {server_args_phase1}")
    server_proc_phase1 = subprocess.Popen(server_args_phase1)
    time.sleep(2)

    # Launch Phase 1 clients
    client_procs_phase1 = []
    for client in clients_cfg:
        client_args = [
            "python", "client.py",
            "--ip", client['ip'],
            "--p", str(client['port']),
            "--cid", str(client['client_id']),
            "--d", client['dataset_name'],
            "--splits", str(client.get('splits', 3))
        ]
        print(f"[{instance_id}] Starting Phase 1 client: {client_args}")
        proc = subprocess.Popen(client_args)
        client_procs_phase1.append(proc)

    # Wait for Phase 1 server to complete
    server_proc_phase1.wait()
    print(f"[{instance_id}] Phase 1 server completed.")
    # Wait for all Phase 1 clients to terminate
    for proc in client_procs_phase1:
        proc.wait()
    print(f"[{instance_id}] All Phase 1 clients terminated.")

    # Wait until the port is fully released before starting Phase 2
    wait_for_port_release(server_cfg['port'], host="127.0.0.1", timeout=30)

    # ---------------- Phase 2: Relaunch server with folder and mode uw ----------------
    print(f"[{instance_id}] Starting Phase 2 (unweighted mode)...")
    server_args_phase2 = [
        "python", "server.py",
        "--p", str(server_cfg['port']),
        "--c", str(server_cfg['num_clients']),
        "--i", str(server_cfg.get('num_iterations', 100)),
        "--f", str(server_cfg.get('force_rounds', 15)),
        "--m", server_cfg.get('model_type', 'SGDC'),
        "--w", server_cfg.get('second_phase_mode', 'uw'),
        "--combs", str(server_cfg.get('combinations', 1)),
        "--lr", str(server_cfg.get('lr', 0.10))
    ]
    print(f"[{instance_id}] Phase 2 server args: {server_args_phase2}")
    server_proc_phase2 = subprocess.Popen(server_args_phase2)
    time.sleep(2)

    # Launch Phase 2 clients
    client_procs_phase2 = []
    for client in clients_cfg:
        client_args = [
            "python", "client.py",
            "--ip", client['ip'],
            "--p", str(client['port']),
            "--cid", str(client['client_id']),
            "--d", client['dataset_name'],
            "--splits", str(client.get('splits', 3))
        ]
        print(f"[{instance_id}] Starting Phase 2 client: {client_args}")
        proc = subprocess.Popen(client_args)
        client_procs_phase2.append(proc)

    # Launch the noFed client for Phase 2
    nofed_args = [
        "python", "noFed_client.py",
        "--c", str(nofed_cfg.get('num_clients', 1)),
        "--d", nofed_cfg['dataset_name']
    ]
    print(f"[{instance_id}] Starting noFed_client: {nofed_args}")
    nofed_proc = subprocess.Popen(nofed_args)

    # Wait for Phase 2 server to complete
    server_proc_phase2.wait()
    print(f"[{instance_id}] Phase 2 server completed.")
    # Wait for all Phase 2 clients to terminate
    for proc in client_procs_phase2:
        proc.wait()
    nofed_proc.wait()
    print(f"[{instance_id}] All Phase 2 clients and noFed_client terminated.")

    # Wait until the port is fully released before starting Phase 2
    wait_for_port_release(server_cfg['port'], host="127.0.0.1", timeout=30)

    # ---------------- Phase 3: Launch server with mode None (No htuning) ----------------
    print(f"[{instance_id}] Starting Phase 3 (None mode)...")
    server_args_phase3 = [
        "python3", "server.py",
        "--p", str(server_cfg['port']),
        "--c", str(server_cfg['num_clients']),
        "--i", str(server_cfg.get('num_iterations', 100)),
        "--f", str(server_cfg.get('force_rounds', 15)),
        "--m", server_cfg.get('model_type', 'SGDC'),
        "--combs", str(server_cfg.get('combinations', 1))
    ]
    print(f"[{instance_id}] Phase 3 server args: {server_args_phase3}")
    server_proc_phase3 = subprocess.Popen(server_args_phase3)
    time.sleep(2)

    # Launch Phase 1 clients
    client_procs_phase3 = []
    for client in clients_cfg:
        client_args = [
            "python", "client.py",
            "--ip", client['ip'],
            "--p", str(client['port']),
            "--cid", str(client['client_id']),
            "--d", client['dataset_name'],
            "--splits", str(client.get('splits', 3))
        ]
        print(f"[{instance_id}] Starting Phase 3 client: {client_args}")
        proc = subprocess.Popen(client_args)
        client_procs_phase3.append(proc)

    # Wait for Phase 1 server to complete
    server_proc_phase3.wait()
    print(f"[{instance_id}] Phase 3 server completed.")
    # Wait for all Phase 1 clients to terminate
    for proc in client_procs_phase3:
        proc.wait()
    print(f"[{instance_id}] All Phase 3 clients terminated.")

    print(f"[{instance_id}] Instance completed.\n")


def main(config_file):
    # Load multi-instance configuration from JSON file
    with open(config_file, 'r') as f:
        multi_config = json.load(f)

    instance_configs = multi_config["instances"]

    # Run each instance sequentially
    for inst_cfg in instance_configs:
        run_instance(inst_cfg)
        # Optionally, wait a few seconds before starting the next instance
        time.sleep(2)
    print("All instances have completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multi_instance_orchestrator.py <multi_instance_config.json>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)
