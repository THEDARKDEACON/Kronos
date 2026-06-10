#!/bin/bash
echo "Syncing Kronos data from Raspberry Pi..."

PI_USER="Deacon"
PI_IP="192.168.100.105"
PI_KRONOS_DIR="/home/Deacon/Kronos"
LOCAL_KRONOS_DIR="/home/gareth-joel/Downloads/Kronos"

# Sync the data/cache directory
rsync -avz --progress \
    $PI_USER@$PI_IP:$PI_KRONOS_DIR/data/cache/ \
    $LOCAL_KRONOS_DIR/data/cache/

# Sync the research directory (for IC history)
rsync -avz --progress \
    $PI_USER@$PI_IP:$PI_KRONOS_DIR/research/ \
    $LOCAL_KRONOS_DIR/research/

echo "Sync complete! You can now view the latest dashboard on your PC."

