# ROSA NVMe-TCP Performance Testing

---

## Expected Performance

| Path                            | Protocol   | Expected Read    | Expected Write   |
|---------------------------------|------------|------------------|------------------|
| Direct local NVMe (internal)    | PCIe       | 6,000+ MB/s      | 4,000+ MB/s      |
| ROSA via LAN (192.168.0.0/24)  | NVMe-TCP   | 1,000–2,500 MB/s | 800–1,500 MB/s   |
| ROSA via dedicated 100G link    | NVMe-TCP   | 8,000–10,000 MB/s| 5,000–8,000 MB/s |
| Dell R750 NFS (20G bond)        | NFS v4.2   | ~580 MB/s        | ~300 MB/s        |
| TrueNAS (10G)                   | NFS        | ~90 MB/s         | ~180 MB/s        |

---

## 1. Block Device Raw Throughput (dd)

```bash
# Sequential write to ROSA storage
sudo dd if=/dev/zero of=/mnt/rosa-storage/speedtest.tmp \
  bs=1M count=8192 oflag=direct conv=fdatasync
# Expect: 800–2,500 MB/s depending on network path

# Sequential read from ROSA storage
sudo dd if=/mnt/rosa-storage/speedtest.tmp of=/dev/null \
  bs=1M iflag=direct
# Expect: 1,000–2,500 MB/s

# Clean up
sudo rm /mnt/rosa-storage/speedtest.tmp
```

---

## 2. fio Benchmark (More Reliable)

```bash
# Install fio if needed
sudo apt install fio -y

# Sequential read
fio --name=seq-read \
  --filename=/mnt/rosa-storage/fio-test \
  --rw=read --bs=1M --direct=1 \
  --size=4G --numjobs=4 --iodepth=32 \
  --runtime=30 --time_based \
  --group_reporting

# Sequential write
fio --name=seq-write \
  --filename=/mnt/rosa-storage/fio-test \
  --rw=write --bs=1M --direct=1 \
  --size=4G --numjobs=4 --iodepth=32 \
  --runtime=30 --time_based \
  --group_reporting

# Random 4K IOPS
fio --name=rand-read \
  --filename=/mnt/rosa-storage/fio-test \
  --rw=randread --bs=4k --direct=1 \
  --size=4G --numjobs=4 --iodepth=128 \
  --runtime=30 --time_based \
  --group_reporting

# Cleanup
sudo rm -f /mnt/rosa-storage/fio-test
```

---

## 3. Network Layer Throughput

Check the actual network bandwidth being used during I/O:

```bash
# On the Spark — watch enp1s0f1np1 (LAN port, 192.168.0.x)
watch -n 1 'cat /sys/class/net/enp1s0f1np1/statistics/rx_bytes'

# Or with iftop
sudo iftop -i enp1s0f1np1 -f "host 192.168.0.100"
```

---

## 4. Latency Check

```bash
# Ping ROSA
ping -c 10 192.168.0.100

# NVMe latency (single-queue depth)
fio --name=latency-test \
  --filename=/mnt/rosa-storage/lat-test \
  --rw=randread --bs=4k --direct=1 \
  --size=1G --numjobs=1 --iodepth=1 \
  --runtime=10 --time_based \
  --group_reporting --lat_percentiles=1
```

---

## 5. Training Data Loading Benchmark

Simulate how fast training data can be streamed:

```bash
# Time reading 10GB of training data sequentially
time dd if=/mnt/rosa-storage/training-benchmarks/train_benchmark.jsonl \
  of=/dev/null bs=1M iflag=direct

# Or with Python (how PyTorch DataLoader will see it)
python3 -c "
import time, os
start = time.time()
with open('/mnt/rosa-storage/training-benchmarks/train_benchmark.jsonl') as f:
    data = f.read()
elapsed = time.time() - start
mb = len(data) / 1e6
print(f'Read {mb:.1f} MB in {elapsed:.2f}s = {mb/elapsed:.0f} MB/s')
"
```

---

## 6. Compare Storage Tiers

Run the same test across all three storage backends:

```bash
for path in /mnt/rosa-storage /mnt/dell-shared /home/sem; do
  echo "=== $path ==="
  sudo dd if=/dev/zero of=${path}/speedtest.tmp \
    bs=1M count=2048 oflag=direct conv=fdatasync 2>&1 | tail -1
  sudo dd if=${path}/speedtest.tmp of=/dev/null \
    bs=1M iflag=direct 2>&1 | tail -1
  sudo rm -f ${path}/speedtest.tmp
done
```

---

## Troubleshooting Slow Performance

**Symptom:** Reads are only ~300 MB/s on ROSA  
**Likely cause:** Traffic going via LAN (192.168.0.x / 1G path) instead of 100G ports  
**Check:**
```bash
# Confirm ROSA traffic is on the right interface
sudo iftop -i enp1s0f1np1   # should be busy
sudo iftop -i enp1s0f0np0   # should be idle (that's the Spark-to-Spark fabric)
```

**Symptom:** Disconnects during training  
**Fix:**
```bash
sudo systemctl restart nvme-rosa-connect.service
sudo mount -a
```
