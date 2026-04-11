# Monitoring Your DGX Spark Cluster

**Know what your cluster is doing** — GPU utilization, temperatures, memory, network throughput, and training job status.

---

## Quick Monitoring (No Setup Required)

### GPU Status (Both Nodes)

```bash
# Single snapshot
nvidia-smi

# Live updating (every 2 seconds)
watch -n 2 nvidia-smi

# Both nodes at once (from spark-dgx-1)
nvidia-smi && echo "--- spark-dgx-2 ---" && ssh sem@10.0.0.2 nvidia-smi
```

### Compact GPU Query

```bash
# One-liner: utilization, memory, temp, power
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits
```

Output: `NVIDIA Graphics Device, 85, 45000, 131072, 62, 120`

### Network Throughput

```bash
# Install if needed: sudo apt install -y iftop
sudo iftop -i enp1s0f0np0   # Watch fabric link traffic during training
```

### Storage

```bash
# ROSA volume usage
df -h | grep rosa

# NVMe health
sudo nvme smart-log /dev/nvme1n1
```

---

## Monitoring Script

A single script that checks everything and can run on a cron.

```bash
#!/bin/bash
# cluster_status.sh — run from spark-dgx-1

echo "=========================================="
echo "DGX Spark Cluster Status — $(date)"
echo "=========================================="

# GPU status — both nodes
for node in "spark-dgx-1:localhost" "spark-dgx-2:10.0.0.2"; do
    name="${node%%:*}"
    host="${node##*:}"
    echo ""
    echo "--- $name ---"

    if [ "$host" = "localhost" ]; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader 2>/dev/null || echo "  GPU: unreachable"
    else
        ssh -o ConnectTimeout=3 sem@$host \
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader" 2>/dev/null || echo "  GPU: unreachable"
    fi
done

# Storage
echo ""
echo "--- Storage ---"
df -h /mnt/rosa-models /mnt/rosa-storage 2>/dev/null | tail -n +2

# Network
echo ""
echo "--- Fabric Link ---"
ip -s link show enp1s0f0np0 2>/dev/null | grep -E "RX:|TX:" | head -2

# EXO status
echo ""
echo "--- EXO Service ---"
systemctl is-active exo-cuda.service 2>/dev/null || echo "not running"

# Uptime
echo ""
echo "--- Uptime ---"
echo "spark-dgx-1: $(uptime -p)"
ssh -o ConnectTimeout=3 sem@10.0.0.2 "echo \"spark-dgx-2: \$(uptime -p)\"" 2>/dev/null || echo "spark-dgx-2: unreachable"

echo ""
echo "=========================================="
```

```bash
chmod +x cluster_status.sh

# Run manually
./cluster_status.sh

# Run every 5 minutes on a cron
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/sem/cluster_status.sh >> /var/log/cluster_status.log 2>&1") | crontab -
```

---

## GPU Metrics Logger

Log GPU stats to a CSV for later analysis or graphing.

```bash
#!/bin/bash
# gpu_logger.sh — logs GPU metrics every 30 seconds
LOGFILE="/var/log/gpu_metrics.csv"

# Write header if file doesn't exist
if [ ! -f "$LOGFILE" ]; then
    echo "timestamp,node,gpu_util_pct,mem_used_mb,mem_total_mb,temp_c,power_w" > "$LOGFILE"
fi

while true; do
    TS=$(date -Iseconds)

    # Local GPU
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | while read line; do
        echo "$TS,spark-dgx-1,$line" >> "$LOGFILE"
    done

    # Remote GPU
    ssh -o ConnectTimeout=3 sem@10.0.0.2 \
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits" 2>/dev/null | while read line; do
        echo "$TS,spark-dgx-2,$line" >> "$LOGFILE"
    done

    sleep 30
done
```

Run in background:
```bash
chmod +x gpu_logger.sh
nohup ./gpu_logger.sh &
```

---

## Prometheus + Grafana (Full Observability)

For a proper dashboard with alerting.

### Install DCGM Exporter (GPU Metrics)

```bash
# On both nodes
docker run -d --gpus all --rm \
    -p 9400:9400 \
    --name dcgm-exporter \
    nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04
```

### Install Node Exporter (System Metrics)

```bash
# On both nodes
sudo apt install -y prometheus-node-exporter
# Runs automatically on port 9100
```

### Install Prometheus (spark-dgx-1)

```bash
sudo apt install -y prometheus

# Add scrape targets
sudo tee -a /etc/prometheus/prometheus.yml << 'EOF'

  - job_name: 'dcgm-spark-1'
    static_configs:
      - targets: ['localhost:9400']

  - job_name: 'dcgm-spark-2'
    static_configs:
      - targets: ['10.0.0.2:9400']

  - job_name: 'node-spark-1'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'node-spark-2'
    static_configs:
      - targets: ['10.0.0.2:9100']
EOF

sudo systemctl restart prometheus
```

### Install Grafana (spark-dgx-1)

```bash
sudo apt install -y grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

Open `http://192.168.0.188:3000` (default login: admin/admin).

Add Prometheus as a data source (`http://localhost:9090`), then import the [NVIDIA DCGM Dashboard](https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/) (ID: 12239).

---

## Thermal Thresholds

DGX Spark runs cool compared to datacenter GPUs, but monitor during sustained training:

| Metric | Normal | Warning | Action |
|---|---|---|---|
| GPU temp | 40-65 C | 70-80 C | Check airflow |
| GPU temp | — | 85+ C | Throttling will start automatically |
| Power draw | 50-120W | 150W+ | Check if expected (large model) |
| Fan noise | Quiet | Audible | Normal under load |

```bash
# Quick thermal check
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader
```

---

## Alert on Failures

Simple email/Slack alert when a training job crashes or a node goes down.

### Systemd Failure Alert

```bash
# Create a notification script
sudo tee /usr/local/bin/notify-failure.sh << 'EOF'
#!/bin/bash
# Send a notification when a systemd service fails
# Customize: replace with your Slack webhook, email, etc.
SERVICE=$1
echo "[$(date)] ALERT: $SERVICE failed on $(hostname)" >> /var/log/cluster_alerts.log

# Optional: Slack webhook
# curl -X POST -H 'Content-type: application/json' \
#     --data "{\"text\":\"ALERT: $SERVICE failed on $(hostname)\"}" \
#     https://hooks.slack.com/services/YOUR/WEBHOOK/URL
EOF
chmod +x /usr/local/bin/notify-failure.sh

# Add to EXO service
sudo mkdir -p /etc/systemd/system/exo-cuda.service.d
sudo tee /etc/systemd/system/exo-cuda.service.d/notify.conf << 'EOF'
[Service]
ExecStopPost=/bin/bash -c 'if [ "$EXIT_STATUS" != "0" ]; then /usr/local/bin/notify-failure.sh exo-cuda; fi'
EOF
sudo systemctl daemon-reload
```

---

## Quick Reference

| What to Monitor | Command | Frequency |
|---|---|---|
| GPU utilization | `nvidia-smi` | During training |
| GPU temperature | `nvidia-smi --query-gpu=temperature.gpu` | Every 5 min |
| Storage usage | `df -h \| grep rosa` | Daily |
| NVMe health | `sudo nvme smart-log /dev/nvme1n1` | Weekly |
| Network link | `ip link show enp1s0f0np0` | On issues |
| EXO service | `systemctl status exo-cuda` | On issues |
| Node reachability | `ping -c1 10.0.0.2` | Every minute |
