# Backup and Recovery

**How to protect your training data, models, and cluster configuration from data loss.**

---

## What to Back Up

| Data | Location | Size | Priority | Backup Method |
|---|---|---|---|---|
| **Trained models** (LoRA adapters) | `/mnt/rosa-models/*/final/` | 50-500 MB each | **Critical** | Copy to NAS or cloud |
| **Merged models** | `/mnt/rosa-models/*/merged/` | 2-140 GB each | High | Copy to NAS |
| **Training data** | `/mnt/rosa-storage/datasets/` | Varies | **Critical** | Copy to NAS + offsite |
| **Cluster configs** | This repo (`configs/`) | <1 MB | **Critical** | Git (already done) |
| **NCCL environment** | `configs/nccl-env.sh` | <1 KB | Critical | Git |
| **Netplan configs** | `/etc/netplan/*.yaml` | <1 KB | Critical | Git + local copy |
| **EXO venv** | `/home/sem/exo-tinygrad-venv/` | ~500 MB | Low | Recreatable |
| **SSH keys** | `~/.ssh/` | <10 KB | High | Secure backup |
| **Systemd services** | `/etc/systemd/system/exo-*` | <1 KB | Medium | Git |

> **Rule:** If it took hours to create (models, data, configs), back it up. If it can be reinstalled in minutes (pip packages, venvs), don't bother.

---

## Backup Strategies

### Strategy 1: Rsync to NAS (Simple)

If you have a NAS on the network (TrueNAS, Synology, etc.):

```bash
#!/bin/bash
# backup_to_nas.sh — run from spark-dgx-1

NAS_TARGET="sem@192.168.0.180:/backups/dgx-spark"
DATE=$(date +%Y%m%d)

echo "=== DGX Spark Backup — $DATE ==="

# Trained models (LoRA adapters are small, back up all of them)
echo "Backing up LoRA adapters..."
rsync -avz --progress /mnt/rosa-models/*/final/ "$NAS_TARGET/models/$DATE/"

# Training data
echo "Backing up training data..."
rsync -avz --progress /mnt/rosa-storage/datasets/ "$NAS_TARGET/datasets/$DATE/"

# Cluster configs
echo "Backing up configs..."
rsync -avz --progress /home/sem/dgx-spark-direct-fabric/configs/ "$NAS_TARGET/configs/"

# SSH keys
echo "Backing up SSH keys..."
rsync -avz --progress ~/.ssh/ "$NAS_TARGET/ssh-keys/"

echo "=== Backup complete ==="
```

### Strategy 2: ROSA Volume Snapshots (If Supported)

The MikroTik RDS2216 may support volume snapshots depending on your RAID configuration. Check RouterOS:

```
# In RouterOS terminal
/disk print
/disk snapshot add name=models-backup-2026-04 volume=llm-models
```

> **Note:** ROSA snapshot support depends on your RAID level and firmware version. Test before relying on it.

### Strategy 3: Cloud Backup (Offsite)

For critical data that can't be recreated:

```bash
# Using rclone (supports S3, GCS, Backblaze B2, etc.)
sudo apt install -y rclone

# Configure (interactive)
rclone config
# Choose your cloud provider, set up credentials

# Sync LoRA adapters to cloud
rclone sync /mnt/rosa-models/ remote:dgx-spark-backups/models/ \
    --include "*/final/**" \
    --progress

# Sync training data
rclone sync /mnt/rosa-storage/datasets/ remote:dgx-spark-backups/datasets/ \
    --progress
```

---

## Automated Backup (Cron)

```bash
# Run backup every night at 2 AM
crontab -e

# Add this line:
0 2 * * * /home/sem/backup_to_nas.sh >> /var/log/dgx-backup.log 2>&1
```

---

## Recovery Procedures

### Scenario 1: Lost a Trained Model

If the LoRA adapter or merged model was deleted:

```bash
# Check NAS backup
ls /mnt/nas/backups/dgx-spark/models/

# Restore specific model
rsync -avz sem@192.168.0.180:/backups/dgx-spark/models/20260401/my-model/ \
    /mnt/rosa-models/my-model/final/
```

### Scenario 2: ROSA Volume Failure

If the ROSA NVMe-TCP storage goes down:

1. **Check NVMe-TCP connection:**
   ```bash
   nvme list-subsys
   sudo systemctl restart nvme-rosa-connect.service
   ```

2. **Remount:**
   ```bash
   sudo mount -a
   df -h | grep rosa
   ```

3. **If the volume is corrupted:**
   ```bash
   # Check filesystem
   sudo fsck.ext4 /dev/nvme1n1

   # If unrecoverable, restore from backup
   sudo mkfs.ext4 -L rosa-models /dev/nvme3n1
   sudo mount /dev/nvme3n1 /mnt/rosa-models
   rsync -avz sem@192.168.0.180:/backups/dgx-spark/models/ /mnt/rosa-models/
   ```

### Scenario 3: Full Node Rebuild

If a DGX Spark needs to be reinstalled from scratch:

1. **Reinstall DGX OS** from NVIDIA recovery image
2. **Apply netplan config:**
   ```bash
   sudo cp configs/netplan-nodeX.yaml /etc/netplan/99-dgx-cluster.yaml
   sudo netplan apply
   ```
3. **Set up SSH keys:**
   ```bash
   # Restore from backup or regenerate
   ssh-keygen -t ed25519
   ssh-copy-id sem@10.0.0.X  # Copy to other node
   ```
4. **Connect ROSA storage:** Follow [Spark Initiator Setup](rosa/02-spark-initiator-setup.md)
5. **Install training dependencies:**
   ```bash
   pip3 install torch --index-url https://download.pytorch.org/whl/cu124
   pip3 install -r training/requirements.txt
   ```
6. **Set up EXO:** Follow [CUDA/Tinygrad Setup](07-exo-cuda-tinygrad.md)
7. **Verify:** `./scripts/verify_network_setup.sh`

---

## What NOT to Back Up

| Don't Back Up | Why |
|---|---|
| Base models (Llama, Qwen) | Re-download from Hugging Face. 70B+ models are huge |
| pip packages / venvs | Recreate with `pip install` |
| Docker images | Re-pull from registry |
| NCCL test binaries | Recompile in 30 seconds |
| `/tmp` or cache dirs | Ephemeral by design |
