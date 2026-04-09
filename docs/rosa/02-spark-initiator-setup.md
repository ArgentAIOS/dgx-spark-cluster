# Spark NVMe-TCP Initiator Setup

**Run on:** Both spark-dgx-1 and spark-dgx-2  
**Result:** ROSA volumes appear as `/dev/nvme1n1` and `/dev/nvme3n1`

---

## 1. Load the nvme-tcp Kernel Module

```bash
# Load immediately
sudo modprobe nvme-tcp

# Persist across reboots
echo "nvme-tcp" | sudo tee /etc/modules-load.d/nvme-tcp.conf

# Verify
lsmod | grep nvme_tcp
```

---

## 2. Discover Available Subsystems

```bash
nvme discover -t tcp -a 192.168.0.100 -s 4420
```

You should see two subsystems: `raid5-data` and `llm-models`.

---

## 3. Connect to NVMe-TCP Targets

```bash
# Connect to the dataset volume
sudo nvme connect -t tcp -n raid5-data -a 192.168.0.100 -s 4420

# Connect to the models volume
sudo nvme connect -t tcp -n llm-models -a 192.168.0.100 -s 4420
```

Verify the new block devices appeared:

```bash
nvme list
# Should show: /dev/nvme1n1 (raid5-data) and /dev/nvme3n1 (llm-models)

lsblk | grep nvme
```

---

## 4. First-Time Only: Format the Volumes

> **Skip this step if the volumes are already formatted — it will erase all data.**

```bash
# Format raid5-data as ext4 (for shared dataset storage)
sudo mkfs.ext4 -L rosa-storage /dev/nvme1n1

# Format llm-models as ext4
sudo mkfs.ext4 -L rosa-models /dev/nvme3n1
```

---

## 5. Create Mount Points and Mount

```bash
sudo mkdir -p /mnt/rosa-storage /mnt/rosa-models

sudo mount /dev/nvme1n1 /mnt/rosa-storage
sudo mount /dev/nvme3n1 /mnt/rosa-models

# Verify
df -h | grep rosa
```

---

## 6. Make Mounts Persistent (fstab)

```bash
sudo nano /etc/fstab
```

Add these two lines (see `configs/fstab-entries.txt`):

```
/dev/nvme1n1  /mnt/rosa-storage  ext4  defaults,noatime,_netdev  0  2
/dev/nvme3n1  /mnt/rosa-models   ext4  defaults,noatime,_netdev  0  2
```

> `_netdev` tells systemd to wait for the network before mounting.  
> `noatime` avoids write amplification on NVMe (improves throughput).

Test:

```bash
sudo umount /mnt/rosa-storage /mnt/rosa-models
sudo mount -a
df -h | grep rosa
```

---

## 7. Auto-Connect at Boot (Systemd Service)

The `nvme connect` commands must run before the fstab mounts happen.
Install the included systemd service:

```bash
sudo cp configs/nvme-rosa-connect.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nvme-rosa-connect.service
sudo systemctl start nvme-rosa-connect.service

# Check status
sudo systemctl status nvme-rosa-connect.service
```

The service file (`configs/nvme-rosa-connect.service`):

```ini
[Unit]
Description=Connect to Rosa NVMe-TCP targets
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/sbin/nvme connect -t tcp -n raid5-data -a 192.168.0.100 -s 4420
ExecStart=/usr/sbin/nvme connect -t tcp -n llm-models -a 192.168.0.100 -s 4420
ExecStop=/usr/sbin/nvme disconnect -n raid5-data
ExecStop=/usr/sbin/nvme disconnect -n llm-models

[Install]
WantedBy=multi-user.target
```

---

## 8. Verify Full Boot Sequence

Reboot and confirm:

```bash
sudo reboot
# ... after reboot ...

# 1. Service ran
sudo systemctl status nvme-rosa-connect.service
# Should show: active (exited)

# 2. Block devices exist
nvme list
lsblk | grep nvme

# 3. Volumes mounted
df -h | grep rosa
mount | grep rosa
```

---

## Quick Reference

```bash
# Check connection status
nvme list-subsys

# Reconnect manually (if disconnected)
sudo systemctl restart nvme-rosa-connect.service

# Disconnect all ROSA volumes
sudo nvme disconnect -n raid5-data
sudo nvme disconnect -n llm-models

# Check your hostnqn (needed for RDS2216 host ACL)
cat /etc/nvme/hostnqn
```
