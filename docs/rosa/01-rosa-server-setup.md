# ROSA Server Setup — MikroTik RDS2216 NVMe-TCP Targets

**Device:** MikroTik RDS2216  
**IP:** 192.168.0.100  
**OS:** RouterOS  
**Ports used:** 2× 100G QSFP28 (to Sparks), 4× 25G SFP28 (LAN/Dell)

---

## 1. Physical Setup

### NVMe Drive Installation

1. Power off the RDS2216
2. Install U.2 NVMe drives in the front bays
3. This cluster uses two logical volumes:
   - **raid5-data** — RAID 5 across multiple NVMe drives (dataset storage)
   - **llm-models** — Dedicated NVMe volume (model weights, ~fast single-drive)
4. Power on

### Network Cabling

| RDS2216 Port | Cable    | Connected To              |
|--------------|----------|---------------------------|
| QSFP28-1     | 100G DAC | spark-dgx-1 (enP2p1s0f0np0 or enp1s0f1np1) |
| QSFP28-2     | 100G DAC | spark-dgx-2               |
| SFP28-1/2    | 25G DAC  | LAN switch / Dell R750    |

> **Note:** The current working setup uses the 192.168.0.0/24 LAN for NVMe-TCP
> (not dedicated 100G links). The 100G ports are available for future upgrade to
> dedicated storage fabric.

---

## 2. RouterOS NVMe-TCP Configuration

Connect to the RDS2216 via SSH or Winbox.

### 2a. Verify NVMe Drives Are Detected

```routeros
/nvme print
```

Expected output — each installed NVMe shows as a device:
```
# MODEL                    SIZE
0 Samsung PM9A3             3840G
1 Samsung PM9A3             3840G
...
```

### 2b. Create Storage Pools (if not already done)

For RAID 5 across multiple drives (the `raid5-data` volume):

```routeros
# Create a RAID 5 pool from drives 0-4
/disk/pool/add name=raid5-data drives=0,1,2,3,4 type=raid5
```

For a dedicated single/mirrored volume (the `llm-models` volume):

```routeros
/disk/pool/add name=llm-models drives=5 type=simple
# Or RAID 1 mirror:
/disk/pool/add name=llm-models drives=5,6 type=mirror
```

Verify pools:

```routeros
/disk/pool/print
```

### 2c. Create NVMe-TCP Subsystems

Each pool is exported as an NVMe subsystem with a unique NQN (NVMe Qualified Name):

```routeros
# Export raid5-data pool as NVMe-TCP subsystem
/nvme-of/subsystem/add \
  name=raid5-data \
  nqn=raid5-data \
  pool=raid5-data

# Export llm-models pool
/nvme-of/subsystem/add \
  name=llm-models \
  nqn=llm-models \
  pool=llm-models
```

### 2d. Configure NVMe-TCP Transport Listener

Bind the NVMe-oF service to the storage network interface on port 4420:

```routeros
# Listen on all interfaces (or specify storage interface IP)
/nvme-of/listener/add \
  transport=tcp \
  address=0.0.0.0 \
  port=4420
```

### 2e. Set Host Access (Optional — Allow All)

By default RouterOS may allow all hosts. To explicitly allow the Spark hosts:

```routeros
# Allow spark-dgx-1
/nvme-of/subsystem/host/add \
  subsystem=raid5-data \
  nqn=nqn.2014-08.org.nvmexpress:uuid:b0042149-1898-4dc6-99ed-c5eb99bf2e06

# Repeat for llm-models and for spark-dgx-2's hostnqn
```

> **Get a Spark's hostnqn:**
> ```bash
> cat /etc/nvme/hostnqn
> # Returns: nqn.2014-08.org.nvmexpress:uuid:<uuid>
> ```

### 2f. Verify Target Is Listening

From any Linux machine on the network:

```bash
nvme discover -t tcp -a 192.168.0.100 -s 4420
```

Expected output:
```
Discovery Log Number of Records 2, Generation counter 1
=====Discovery Log Entry 0======
trtype:  tcp
adrfam:  ipv4
subtype: nvme subsystem
treq:    not specified
portid:  0
trsvcid: 4420
subnqn:  raid5-data
traddr:  192.168.0.100

=====Discovery Log Entry 1======
...
subnqn:  llm-models
traddr:  192.168.0.100
```

---

## 3. Network Configuration on RDS2216

Assign a static IP to the storage interface:

```routeros
/ip/address/add address=192.168.0.100/24 interface=<storage-interface>
```

Verify routing to the Sparks:

```routeros
/ping 192.168.0.188   # spark-dgx-1
/ping 192.168.0.218   # spark-dgx-2
```

---

## 4. Useful RouterOS Diagnostics

```routeros
# List all NVMe devices
/nvme print

# List storage pools
/disk/pool/print

# List NVMe-oF subsystems
/nvme-of/subsystem/print

# List active NVMe-oF connections (connected hosts)
/nvme-of/connection/print

# Check interface statistics
/interface/print stats
```
