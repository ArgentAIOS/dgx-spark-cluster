#!/usr/bin/env python3
"""
Main benchmark orchestration script.
Runs training benchmarks across different configurations and collects results.
"""

import argparse
import shlex
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


class BenchmarkOrchestrator:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config["results_dir"]).expanduser()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_dir = Path(__file__).parent
        self.all_results = []
    
    def check_ssh_connection(self, host):
        """Check if we can SSH to a host."""
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", f"{self.config['ssh']['user']}@{host}", "echo", "ok"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def cleanup_stale_processes(self, nodes, master_port="29500"):
        """Kill any stale PyTorch processes that might be holding ports."""
        print("Cleaning up stale processes...")
        for node in nodes:
            try:
                if node == "localhost":
                    # Local cleanup
                    subprocess.run(
                        f"lsof -ti :{master_port} | xargs -r kill -9",
                        shell=True,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    # Remote cleanup via SSH
                    subprocess.run(
                        ["ssh", f"{self.config['ssh']['user']}@{node}", 
                         f"lsof -ti :{master_port} | xargs -r kill -9"],
                        stderr=subprocess.DEVNULL
                    )
                print(f"  ✅ Cleaned up {node}")
            except Exception as e:
                # Ignore errors if no processes found
                pass
    
    def run_single_node(self, scenario_name, scenario_config, node_host=None):
        """Run single-node training benchmark."""
        print(f"\n{'='*70}")
        print(f"Running: {scenario_config['name']}")
        print(f"{'='*70}")
        
        # Determine output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.results_dir / f"{scenario_name}_{timestamp}"
        
        # Build command
        train_params = self.config["training"]
        cmd = [
            "python3",
            str(self.benchmark_dir / "benchmark_train.py"),
            "--model-name", train_params["model_name"],
            "--train-data", scenario_config["train_data"],
            "--val-data", scenario_config["val_data"],
            "--output-dir", str(output_dir),
            "--max-steps", str(train_params["max_steps"]),
            "--batch-size", str(train_params["batch_size"]),
            "--gradient-accumulation", str(train_params["gradient_accumulation"]),
            "--learning-rate", str(train_params["learning_rate"]),
            "--lora-rank", str(train_params["lora_rank"]),
            "--lora-alpha", str(train_params["lora_alpha"]),
            "--max-seq-length", str(train_params["max_seq_length"]),
        ]
        
        # Run locally or via SSH
        if node_host and node_host != "localhost":
            # Create output directory on remote host
            subprocess.run(
                ["ssh", f"{self.config['ssh']['user']}@{node_host}", "mkdir", "-p", str(output_dir)],
                check=True
            )
            
            # Run via SSH
            ssh_cmd = ["ssh", f"{self.config['ssh']['user']}@{node_host}"] + cmd
            print(f"Running on {node_host}...")
            result = subprocess.run(ssh_cmd)
        else:
            # Run locally
            output_dir.mkdir(parents=True, exist_ok=True)
            print("Running locally...")
            result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"❌ Benchmark failed with exit code {result.returncode}")
            return None
        
        # Collect metrics
        metrics_file = output_dir / "benchmark_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            metrics["scenario_name"] = scenario_name
            self.all_results.append(metrics)
            print(f"✅ Benchmark completed successfully")
            return metrics
        else:
            print(f"⚠️  Metrics file not found: {metrics_file}")
            return None
    
    def run_distributed(self, scenario_name, scenario_config):
        """Run distributed (2-node) training benchmark."""
        print(f"\n{'='*70}")
        print(f"Running: {scenario_config['name']}")
        print(f"{'='*70}")
        
        node0 = scenario_config["node0"]
        node1 = scenario_config["node1"]
        master_addr = scenario_config["master_addr"]
        master_port = "29500"
        
        # Clean up any stale processes first
        self.cleanup_stale_processes([node0, node1], master_port)
        
        # Check SSH connections
        print("Checking SSH connections...")
        if not self.check_ssh_connection(node1):
            print(f"❌ Cannot connect to {node1}")
            return None
        print(f"✅ Connected to {node1}")
        
        # Determine output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.results_dir / f"{scenario_name}_{timestamp}"
        
        # Create output directories on both nodes
        for node in [node0, node1]:
            if node == "localhost":
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                subprocess.run(
                    ["ssh", f"{self.config['ssh']['user']}@{node}", "mkdir", "-p", str(output_dir)],
                    check=True
                )
        
        # Build base command
        train_params = self.config["training"]
        base_cmd = [
            "python3",
            str(self.benchmark_dir / "benchmark_train.py"),
            "--model-name", train_params["model_name"],
            "--train-data", scenario_config["train_data"],
            "--val-data", scenario_config["val_data"],
            "--output-dir", str(output_dir),
            "--max-steps", str(train_params["max_steps"]),
            "--batch-size", str(train_params["batch_size"]),
            "--gradient-accumulation", str(train_params["gradient_accumulation"]),
            "--learning-rate", str(train_params["learning_rate"]),
            "--lora-rank", str(train_params["lora_rank"]),
            "--lora-alpha", str(train_params["lora_alpha"]),
            "--max-seq-length", str(train_params["max_seq_length"]),
        ]
        
        # Build torchrun commands for each node
        node0_cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--nnodes=2",
            "--node_rank=0",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
        ] + base_cmd
        
        node1_cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--nnodes=2",
            "--node_rank=1",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
        ] + base_cmd
        
        # Build SSH command that activates conda environment
        conda_activate = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal-injury-llm && "
        node1_cmd_str = " ".join(shlex.quote(arg) for arg in node1_cmd)
        ssh_node1_cmd = ["ssh", f"{self.config['ssh']['user']}@{node1}", "bash", "-c", f"{conda_activate}{node1_cmd_str}"]
        
        # Start node 0 (master) locally in background
        print(f"Starting training on {node0} (node 0 - master)...")
        node0_process = subprocess.Popen(node0_cmd)
        
        # Give node 0 time to initialize and start listening
        time.sleep(10)
        
        # Start node 1 via SSH
        print(f"Starting training on {node1} (node 1)...")
        node1_log = open(f"/tmp/node1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w")
        node1_process = subprocess.Popen(
            ssh_node1_cmd,
            stdout=node1_log,
            stderr=subprocess.STDOUT,
        )
        print(f"Node 1 output being written to: {node1_log.name}")
        
        # Wait for both to complete
        print("Waiting for training to complete...")
        node0_process.wait()
        node1_process.wait()
        
        node0_result = type('obj', (object,), {'returncode': node0_process.returncode})()
        
        if node0_result.returncode != 0 or node1_process.returncode != 0:
            print(f"❌ Distributed benchmark failed")
            print(f"   Node 0 exit code: {node0_result.returncode}")
            print(f"   Node 1 exit code: {node1_process.returncode}")
            return None
        
        # Collect metrics from node 0
        metrics_file = output_dir / "benchmark_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            metrics["scenario_name"] = scenario_name
            self.all_results.append(metrics)
            print(f"✅ Distributed benchmark completed successfully")
            return metrics
        else:
            print(f"⚠️  Metrics file not found: {metrics_file}")
            return None
    
    def run_scenario(self, scenario_name):
        """Run a specific scenario."""
        if scenario_name not in self.config["scenarios"]:
            print(f"❌ Unknown scenario: {scenario_name}")
            return None
        
        scenario_config = self.config["scenarios"][scenario_name]
        
        if scenario_config["nodes"] == 1:
            return self.run_single_node(scenario_name, scenario_config)
        elif scenario_config["nodes"] == 2:
            return self.run_distributed(scenario_name, scenario_config)
        else:
            print(f"❌ Unsupported node count: {scenario_config['nodes']}")
            return None
    
    def generate_report(self):
        """Generate comparison report from all results."""
        if not self.all_results:
            print("\n⚠️  No results to report")
            return
        
        print("\n" + "="*70)
        print("BENCHMARK COMPARISON REPORT")
        print("="*70)
        print(f"{'Scenario':<30} {'Time (s)':<12} {'Steps/s':<12} {'Samples/s':<12}")
        print("-"*70)
        
        for result in sorted(self.all_results, key=lambda x: x["elapsed_seconds"]):
            print(f"{result['scenario_name']:<30} "
                  f"{result['elapsed_seconds']:<12.2f} "
                  f"{result['steps_per_second']:<12.4f} "
                  f"{result['samples_per_second']:<12.2f}")
        
        print("="*70)
        
        # Save full results
        report_file = self.results_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\nFull report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Run training benchmarks")
    parser.add_argument(
        "--scenario",
        help="Run specific scenario (or 'all' for all scenarios)",
        default="all"
    )
    parser.add_argument(
        "--config",
        default="~/training-benchmarks/configs.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios"
    )
    
    args = parser.parse_args()
    
    config_file = Path(args.config).expanduser()
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    orchestrator = BenchmarkOrchestrator(config_file)
    
    if args.list:
        print("\nAvailable scenarios:")
        for name, config in orchestrator.config["scenarios"].items():
            print(f"  {name:<25} - {config['name']}")
        sys.exit(0)
    
    # Run scenarios
    if args.scenario == "all":
        scenarios = list(orchestrator.config["scenarios"].keys())
    else:
        scenarios = [args.scenario]
    
    print(f"\nRunning {len(scenarios)} scenario(s)...")
    
    for scenario in scenarios:
        try:
            orchestrator.run_scenario(scenario)
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error running scenario '{scenario}': {e}")
            import traceback
            traceback.print_exc()
    
    # Generate final report
    orchestrator.generate_report()


if __name__ == "__main__":
    main()
