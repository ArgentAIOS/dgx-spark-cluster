# Cluster Overview — Full Topology

## Hardware Inventory

| System | Role | GPU | RAM | OS |
|---|---|---|---|---|
| spark-dgx-1 | Training master (rank 0) | NVIDIA GB10 (Grace Blackwell) | Unified | DGX Spark OS 7.5.0 |
| spark-dgx-2 | Training worker (rank 1) | NVIDIA GB10 (Grace Blackwell) | Unified | DGX Spark OS 7.5.0 |
| MikroTik RDS2216 (ROSA) | NVMe-TCP storage server | — | 32 GB DDR4 | RouterOS |
| Dell PowerEdge R750 | Warm/shared storage | — | — | Ubuntu |
| TrueNAS | Archive/cold storage | — | — | TrueNAS |
| MikroTik CRS812 | 10G management switch | — | — | RouterOS |

---

## Full Network Topology

```
                        ┌─────────────────────────────────────────┐
                        │      MikroTik RDS2216 (ROSA)            │
                        │      192.168.0.100                      │
                        │      20× U.2 NVMe  |  2×100G  4×25G    │
                        └────────┬──────────────────┬─────────────┘
                     100G QSFP28 │                  │ 100G QSFP28
                     (future)    │                  │ (future)
                                 │   [current path] │
                    25G/LAN ─────┴──────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼───────────┐       ┌─────────▼───────────┐
│    spark-dgx-1      │       │    spark-dgx-2       │
│    Rank 0 / Master  │       │    Rank 1 / Worker   │
│                     │       │                      │
│  enp1s0f0np0        │       │  enp1s0f0np0         │
│  mlx5_0  10.0.0.1   │◄─────►│  mlx5_0  10.0.0.2   │
│  200Gb direct fabric│       │  200Gb direct fabric │
│                     │       │                      │
│  enp1s0f1np1        │       │  enp1s0f1np1         │
│  192.168.0.188/24   │       │  192.168.0.218/24    │
│  LAN / default gw   │       │  LAN / default gw    │
│                     │       │                      │
│  enP7s7             │       │                      │
│  192.168.0.110/24   │       │                      │
│  Management (10G)   │       │                      │
└─────────────────────┘       └──────────────────────┘
          │                             │
          └──────────────┬──────────────┘
                         │ 192.168.0.0/24 LAN
               ┌─────────▼─────────┐
               │  MikroTik CRS812  │
               │  Management switch│
               │  10G / 1G         │
               └─────────┬─────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
┌─────────▼──────┐  ┌────▼───────┐  ┌──▼──────────┐
│  Dell R750     │  │  TrueNAS   │  │  Gateway     │
│  192.168.0.98  │  │ 192.168.0  │  │  192.168.0.1 │
│  NFS /data     │  │ .180       │  │              │
│  50G bonded    │  │ NFS        │  │              │
└────────────────┘  └────────────┘  └─────────────┘
```

---

## IP Address Reference

| Host | Interface | IP | Speed | Use |
|---|---|---|---|---|
| spark-dgx-1 | enp1s0f0np0 (mlx5_0) | 10.0.0.1/24 | 200G | Direct fabric (NCCL) |
| spark-dgx-1 | enp1s0f1np1 (mlx5_1) | 192.168.0.188/24 | 200G | LAN / storage |
| spark-dgx-1 | enP7s7 | 192.168.0.110/24 | 10G | Management |
| spark-dgx-1 | tailscale0 | 100.122.26.9/32 | — | Remote access |
| spark-dgx-2 | enp1s0f0np0 (mlx5_0) | 10.0.0.2/24 | 200G | Direct fabric (NCCL) |
| spark-dgx-2 | enp1s0f1np1 (mlx5_1) | 192.168.0.218/24 | 200G | LAN / storage |
| ROSA (RDS2216) | — | 192.168.0.100 | 100G | NVMe-TCP storage |
| Dell R750 | bond0 | 192.168.0.98 | 50G | NFS shared storage |
| TrueNAS | — | 192.168.0.180 | 10G | NFS archive |
| Gateway | — | 192.168.0.1 | — | Internet |

---

## Storage Mount Reference (per Spark)

| Mount Point | Source | Protocol | Speed | Use |
|---|---|---|---|---|
| /mnt/rosa-storage | ROSA /dev/nvme1n1 | NVMe-TCP | ~2 GB/s (LAN) | Hot datasets |
| /mnt/rosa-models | ROSA /dev/nvme3n1 | NVMe-TCP | ~2 GB/s (LAN) | Model weights |
| /mnt/dell-shared | 192.168.0.98:/data/shared | NFS v4.2 | ~580 MB/s | Working data |
| /mnt/nas | 192.168.0.180:/mnt/Servers/data | NFS | ~90 MB/s | Archives |
| /home/sem | Local Samsung PM9E1 4TB | PCIe NVMe | 6+ GB/s | OS, conda envs |

---

## NCCL Traffic Flow

During distributed training, gradient synchronisation happens **only** over the direct fabric:

```
spark-dgx-1 GPU                              spark-dgx-2 GPU
      │                                             │
      │  enp1s0f0np0 / mlx5_0 / 10.0.0.1          │  enp1s0f0np0 / mlx5_0 / 10.0.0.2
      └──────────────── 200 Gb/s RoCEv2 ───────────┘
                    NCCL allreduce (Ring)
                    ~18-23 GB/s depending on GDR mode
```

Training **data** is read from `/mnt/rosa-storage` (or `/mnt/dell-shared`) over LAN —
this is independent of the NCCL fabric.

---

## Quick Health Check

```bash
# Fabric
ping -c 3 -I enp1s0f0np0 10.0.0.2

# Storage
df -h | grep -E "rosa|dell|nas"
sudo systemctl status nvme-rosa-connect

# GPU
nvidia-smi

# NCCL smoke test
./scripts/distributed_train.sh 0 training/validate_distributed.py  # spark-dgx-1
./scripts/distributed_train.sh 1 training/validate_distributed.py  # spark-dgx-2
```
