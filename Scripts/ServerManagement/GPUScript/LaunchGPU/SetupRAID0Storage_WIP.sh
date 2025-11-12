#!/bin/bash

# RAID 0 Storage Setup Script
# This script creates a RAID 0 array from two disks for balanced read/write performance
# WARNING: This will DESTROY all data on the specified disks!

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default mount point
MOUNT_POINT="/mnt/model-disk"

echo "=================================================="
echo "      RAID 0 Storage Setup Script"
echo "=================================================="
echo ""
echo -e "${YELLOW}WARNING: This script will DESTROY all data on the specified disks!${NC}"
echo -e "${YELLOW}RAID 0 provides better performance but NO redundancy.${NC}"
echo -e "${YELLOW}If one disk fails, ALL data will be lost!${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Check if mdadm is installed
if ! command -v mdadm &> /dev/null; then
    echo "Installing mdadm (RAID management tool)..."
    apt-get update
    apt-get install -y mdadm
fi

# Early check: See if RAID 0 is already properly configured
echo "Checking for existing RAID 0 configuration..."
echo ""

if [ -b "/dev/md0" ]; then
    RAID_LEVEL=$(mdadm --detail /dev/md0 2>/dev/null | grep "Raid Level" | awk '{print $4}')
    RAID_DEVICES=$(mdadm --detail /dev/md0 2>/dev/null | grep "Raid Devices" | awk '{print $4}')

    if [ "$RAID_LEVEL" == "raid0" ] && [ "$RAID_DEVICES" == "2" ]; then
        CURRENT_MOUNT=$(lsblk -lno MOUNTPOINT /dev/md0 | grep -v "^$")

        # Check if mounted at the target mount point
        if [ "$CURRENT_MOUNT" == "$MOUNT_POINT" ]; then
            echo -e "${GREEN}✅ RAID 0 array is already configured and mounted at $CURRENT_MOUNT!${NC}"
            echo ""
            echo "RAID Array Details:"
            mdadm --detail /dev/md0 | grep -E "Raid Level|Raid Devices|Array Size|State"
            echo ""
            echo "Mount Information:"
            df -h "$CURRENT_MOUNT"
            echo ""
            echo "Component Disks:"
            mdadm --detail /dev/md0 | grep -E "\/dev\/(sd[a-z]|nvme[0-9]n[0-9])"
            echo ""
            echo -e "${GREEN}✅ Setup is already complete. Nothing to do!${NC}"
            echo ""
            echo "You can safely run this script again if needed."
            exit 0
        fi
    fi
fi

echo "No existing RAID 0 setup found or needs reconfiguration."
echo ""

# Function to show disk information
show_disk_info() {
    local disk=$1
    echo "  Device: $disk"
    if [ -b "$disk" ]; then
        lsblk -o NAME,SIZE,TYPE,MOUNTPOINT "$disk" 2>/dev/null || true
        echo "  Model: $(lsblk -dno MODEL "$disk" 2>/dev/null || echo 'Unknown')"
    else
        echo -e "  ${RED}Not found or not a block device${NC}"
    fi
}

# Function to convert size to bytes for comparison
size_to_bytes() {
    local size=$1
    local number=$(echo "$size" | grep -oE '[0-9.]+')
    local unit=$(echo "$size" | grep -oE '[A-Z]+')

    case "$unit" in
        T|TB) echo "$(echo "$number * 1099511627776" | bc | cut -d. -f1)" ;;
        G|GB) echo "$(echo "$number * 1073741824" | bc | cut -d. -f1)" ;;
        M|MB) echo "$(echo "$number * 1048576" | bc | cut -d. -f1)" ;;
        K|KB) echo "$(echo "$number * 1024" | bc | cut -d. -f1)" ;;
        *) echo "0" ;;
    esac
}

# Function to get disks larger than 1TB
get_large_disks() {
    local min_size_bytes=1099511627776  # 1TB in bytes
    local -a large_disks=()

    # Get all block devices that are disks (not partitions)
    while IFS= read -r line; do
        local disk=$(echo "$line" | awk '{print $1}')
        local size=$(echo "$line" | awk '{print $2}')
        local type=$(echo "$line" | awk '{print $3}')

        # Only process actual disks, not partitions or loops
        if [ "$type" == "disk" ]; then
            local size_bytes=$(size_to_bytes "$size")
            if [ "$size_bytes" -ge "$min_size_bytes" ]; then
                large_disks+=("/dev/$disk")
            fi
        fi
    done < <(lsblk -ndo NAME,SIZE,TYPE | grep disk)

    echo "${large_disks[@]}"
}

# Function to display disk selection menu
select_disk() {
    local prompt=$1
    shift
    local available_disks=("$@")
    local count=${#available_disks[@]}

    if [ $count -eq 0 ]; then
        echo -e "${RED}Error: No disks available for selection${NC}"
        exit 1
    fi

    echo "$prompt"
    echo ""

    local index=1
    for disk in "${available_disks[@]}"; do
        local size=$(lsblk -dno SIZE "$disk" 2>/dev/null || echo "Unknown")
        local model=$(lsblk -dno MODEL "$disk" 2>/dev/null || echo "Unknown")
        local mount=$(lsblk -lno MOUNTPOINT "$disk" 2>/dev/null | grep -v "^$" | head -1 || echo "Not mounted")

        printf "%2d) %-15s  Size: %-8s  Model: %-20s  Mount: %s\n" "$index" "$disk" "$size" "$model" "$mount"
        ((index++))
    done
    echo ""

    local selection=0
    while true; do
        read -p "Enter number (1-$count): " selection
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le $count ]; then
            echo "${available_disks[$((selection-1))]}"
            return 0
        else
            echo -e "${RED}Invalid selection. Please enter a number between 1 and $count${NC}"
        fi
    done
}

# Parse command line arguments or prompt for input
if [ $# -eq 2 ]; then
    DISK1="$1"
    DISK2="$2"
else
    echo "Scanning for disks larger than 1TB..."
    echo ""

    # Get all large disks
    LARGE_DISKS=($(get_large_disks))

    if [ ${#LARGE_DISKS[@]} -lt 2 ]; then
        echo -e "${RED}Error: At least 2 disks larger than 1TB are required for RAID 0${NC}"
        echo ""
        echo "All available disks:"
        lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL | grep -E "disk|NAME"
        exit 1
    fi

    echo -e "${GREEN}Found ${#LARGE_DISKS[@]} disk(s) larger than 1TB:${NC}"
    echo ""

    # Select first disk
    DISK1=$(select_disk "Select the FIRST disk for RAID 0:" "${LARGE_DISKS[@]}")
    echo -e "${GREEN}Selected: $DISK1${NC}"
    echo ""

    # Remove selected disk from available options
    REMAINING_DISKS=()
    for disk in "${LARGE_DISKS[@]}"; do
        if [ "$disk" != "$DISK1" ]; then
            REMAINING_DISKS+=("$disk")
        fi
    done

    # Select second disk
    DISK2=$(select_disk "Select the SECOND disk for RAID 0:" "${REMAINING_DISKS[@]}")
    echo -e "${GREEN}Selected: $DISK2${NC}"
    echo ""
fi

# Strip partition numbers if provided (we need the base disk)
DISK1=$(echo "$DISK1" | sed 's/p\?[0-9]*$//')
DISK2=$(echo "$DISK2" | sed 's/p\?[0-9]*$//')

# Validate inputs
if [ "$DISK1" == "$DISK2" ]; then
    echo -e "${RED}Error: Both disks cannot be the same device!${NC}"
    exit 1
fi

if [ ! -b "$DISK1" ]; then
    echo -e "${RED}Error: $DISK1 is not a valid block device!${NC}"
    exit 1
fi

if [ ! -b "$DISK2" ]; then
    echo -e "${RED}Error: $DISK2 is not a valid block device!${NC}"
    exit 1
fi

# Show disk information
echo ""
echo "Selected disks:"
echo "----------------------------------------"
echo "Disk 1:"
show_disk_info "$DISK1"
echo ""
echo "Disk 2:"
show_disk_info "$DISK2"
echo "----------------------------------------"
echo ""

# Check if disks are currently mounted
MOUNTED_PARTITIONS=$(lsblk -lno MOUNTPOINT "$DISK1" "$DISK2" 2>/dev/null | grep -v "^$" || true)
if [ -n "$MOUNTED_PARTITIONS" ]; then
    echo -e "${YELLOW}Warning: One or more partitions are currently mounted:${NC}"
    echo "$MOUNTED_PARTITIONS"
    echo ""
fi

# Final confirmation
echo -e "${RED}⚠️  WARNING: This will PERMANENTLY DELETE all data on:${NC}"
echo -e "${RED}   - $DISK1${NC}"
echo -e "${RED}   - $DISK2${NC}"
echo ""
read -p "Type 'YES' in capital letters to continue: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""

# Check if RAID 0 is already configured correctly
echo "Checking for existing RAID 0 configuration..."
echo ""

EXISTING_RAID=""
SKIP_SETUP=false

# Check if /dev/md0 exists and is a RAID 0 array
if [ -b "/dev/md0" ]; then
    RAID_LEVEL=$(mdadm --detail /dev/md0 2>/dev/null | grep "Raid Level" | awk '{print $4}')
    RAID_DEVICES=$(mdadm --detail /dev/md0 2>/dev/null | grep "Raid Devices" | awk '{print $4}')

    if [ "$RAID_LEVEL" == "raid0" ] && [ "$RAID_DEVICES" == "2" ]; then
        echo "Found existing RAID 0 array: /dev/md0"

        # Get the component disks
        COMPONENT_DISKS=$(mdadm --detail /dev/md0 2>/dev/null | grep -E "\/dev\/(sd[a-z]|nvme[0-9]n[0-9])" | awk '{print $NF}')

        # Check if the RAID uses our specified disks
        USES_DISK1=false
        USES_DISK2=false

        for component in $COMPONENT_DISKS; do
            # Strip partition number to get base disk
            BASE_DISK=$(echo "$component" | sed 's/p\?[0-9]*$//')
            if [ "$BASE_DISK" == "$DISK1" ]; then
                USES_DISK1=true
            fi
            if [ "$BASE_DISK" == "$DISK2" ]; then
                USES_DISK2=true
            fi
        done

        if [ "$USES_DISK1" == "true" ] && [ "$USES_DISK2" == "true" ]; then
            echo -e "${GREEN}✅ RAID 0 array already exists with the specified disks!${NC}"
            echo ""
            mdadm --detail /dev/md0
            echo ""

            # Check if it's mounted at the correct location
            CURRENT_MOUNT=$(lsblk -lno MOUNTPOINT /dev/md0 | grep -v "^$")

            if [ "$CURRENT_MOUNT" == "$MOUNT_POINT" ]; then
                echo -e "${GREEN}✅ RAID 0 array is already mounted at $MOUNT_POINT${NC}"
                echo ""
                df -h "$MOUNT_POINT"
                echo ""
                echo "Setup is already complete. Nothing to do!"
                SKIP_SETUP=true
            else
                echo -e "${YELLOW}⚠️  RAID 0 array exists but not mounted at $MOUNT_POINT${NC}"
                if [ -n "$CURRENT_MOUNT" ]; then
                    echo "   Currently mounted at: $CURRENT_MOUNT"
                else
                    echo "   Not currently mounted"
                fi
                echo ""
                read -p "Would you like to remount it at $MOUNT_POINT? (y/n): " REMOUNT
                if [ "$REMOUNT" == "y" ] || [ "$REMOUNT" == "Y" ]; then
                    # Unmount if currently mounted elsewhere
                    if [ -n "$CURRENT_MOUNT" ]; then
                        echo "Unmounting from $CURRENT_MOUNT..."
                        umount /dev/md0
                    fi

                    # Create mount point and mount
                    mkdir -p "$MOUNT_POINT"
                    mount /dev/md0 "$MOUNT_POINT"

                    # Update fstab
                    RAID_UUID=$(blkid -s UUID -o value /dev/md0)
                    cp /etc/fstab /etc/fstab.backup.$(date +%Y%m%d_%H%M%S)
                    sed -i "\|$MOUNT_POINT|d" /etc/fstab
                    echo "UUID=$RAID_UUID $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab

                    echo -e "${GREEN}✅ Remounted successfully!${NC}"
                    df -h "$MOUNT_POINT"
                fi
                SKIP_SETUP=true
            fi
        else
            echo -e "${YELLOW}⚠️  RAID 0 array exists but uses different disks${NC}"
            echo "   Existing array uses:"
            for component in $COMPONENT_DISKS; do
                echo "     - $component"
            done
            echo "   You specified: $DISK1 and $DISK2"
            echo ""
            read -p "Do you want to DESTROY the existing array and create a new one? (y/n): " DESTROY
            if [ "$DESTROY" != "y" ] && [ "$DESTROY" != "Y" ]; then
                echo "Operation cancelled."
                exit 0
            fi
            echo ""
            echo "Will destroy existing array and create new one..."
            # Stop and remove existing array
            umount /dev/md0 2>/dev/null || true
            mdadm --stop /dev/md0
            mdadm --zero-superblock $(echo "$COMPONENT_DISKS" | xargs) 2>/dev/null || true
        fi
    fi
fi

if [ "$SKIP_SETUP" == "true" ]; then
    exit 0
fi

echo "Starting RAID 0 setup..."
echo ""

# Step 1: Unmount any mounted partitions
echo "1️⃣  Unmounting any mounted partitions..."
for disk in "$DISK1" "$DISK2"; do
    PARTITIONS=$(lsblk -lno NAME "$disk" | tail -n +2)
    for part in $PARTITIONS; do
        MOUNT_PATH=$(lsblk -lno MOUNTPOINT "/dev/$part" 2>/dev/null || true)
        if [ -n "$MOUNT_PATH" ]; then
            echo "   Unmounting /dev/$part from $MOUNT_PATH"
            umount "/dev/$part" 2>/dev/null || true
        fi
    done
done
echo "   ✅ Done"
echo ""

# Step 2: Remove existing RAID arrays if any
echo "2️⃣  Checking for existing RAID arrays..."
for disk in "$DISK1" "$DISK2"; do
    PARTITIONS=$(lsblk -lno NAME "$disk" | tail -n +2)
    for part in $PARTITIONS; do
        if mdadm --query "/dev/$part" &>/dev/null; then
            echo "   Removing /dev/$part from RAID array..."
            mdadm --stop "/dev/$part" 2>/dev/null || true
            mdadm --zero-superblock "/dev/$part" 2>/dev/null || true
        fi
    done
done
echo "   ✅ Done"
echo ""

# Step 3: Wipe existing partitions
echo "3️⃣  Wiping existing partition tables..."
wipefs -a "$DISK1" 2>/dev/null || true
wipefs -a "$DISK2" 2>/dev/null || true
echo "   ✅ Done"
echo ""

# Step 4: Create new partition tables
echo "4️⃣  Creating new GPT partition tables..."
parted -s "$DISK1" mklabel gpt
parted -s "$DISK2" mklabel gpt
echo "   ✅ Done"
echo ""

# Step 5: Create partitions
echo "5️⃣  Creating partitions..."
parted -s "$DISK1" mkpart primary 0% 100%
parted -s "$DISK2" mkpart primary 0% 100%
parted -s "$DISK1" set 1 raid on
parted -s "$DISK2" set 1 raid on
echo "   ✅ Done"
echo ""

# Determine partition names (nvme uses p1, others use 1)
if [[ "$DISK1" == *"nvme"* ]]; then
    PART1="${DISK1}p1"
else
    PART1="${DISK1}1"
fi

if [[ "$DISK2" == *"nvme"* ]]; then
    PART2="${DISK2}p1"
else
    PART2="${DISK2}1"
fi

# Wait for partitions to be created
echo "   Waiting for partitions to be available..."
sleep 2
partprobe "$DISK1" "$DISK2"
sleep 2

# Step 6: Create RAID 0 array
echo "6️⃣  Creating RAID 0 array..."
RAID_DEVICE="/dev/md0"

# Stop any existing md0
mdadm --stop "$RAID_DEVICE" 2>/dev/null || true

# Create the RAID array
mdadm --create "$RAID_DEVICE" \
    --level=0 \
    --raid-devices=2 \
    "$PART1" "$PART2"

echo "   ✅ RAID 0 array created at $RAID_DEVICE"
echo ""

# Step 7: Create filesystem
echo "7️⃣  Creating ext4 filesystem..."
mkfs.ext4 -F "$RAID_DEVICE"
echo "   ✅ Done"
echo ""

# Step 8: Create mount point
echo "8️⃣  Creating mount point at $MOUNT_POINT..."
mkdir -p "$MOUNT_POINT"
echo "   ✅ Done"
echo ""

# Step 9: Mount the RAID array
echo "9️⃣  Mounting RAID array..."
mount "$RAID_DEVICE" "$MOUNT_POINT"
echo "   ✅ Done"
echo ""

# Step 10: Update /etc/fstab for persistent mounting
echo "🔟 Updating /etc/fstab for automatic mounting..."

# Get UUID of the RAID device
RAID_UUID=$(blkid -s UUID -o value "$RAID_DEVICE")

# Backup fstab
cp /etc/fstab /etc/fstab.backup.$(date +%Y%m%d_%H%M%S)

# Remove old entries for this mount point
sed -i "\|$MOUNT_POINT|d" /etc/fstab

# Add new entry
echo "UUID=$RAID_UUID $MOUNT_POINT ext4 defaults 0 2" >> /etc/fstab
echo "   ✅ /etc/fstab updated"
echo ""

# Step 11: Save RAID configuration
echo "1️⃣1️⃣  Saving RAID configuration..."
mdadm --detail --scan >> /etc/mdadm/mdadm.conf
update-initramfs -u
echo "   ✅ Done"
echo ""

# Final verification
echo "=================================================="
echo "✅ RAID 0 Setup Complete!"
echo "=================================================="
echo ""
echo "RAID Array Details:"
mdadm --detail "$RAID_DEVICE"
echo ""
echo "Filesystem Information:"
df -h "$MOUNT_POINT"
echo ""
echo "Mount Point: $MOUNT_POINT"
echo ""
echo -e "${GREEN}Your RAID 0 array is now ready to use!${NC}"
echo ""
echo "Performance Tips:"
echo "  • RAID 0 stripes data across both disks for better performance"
echo "  • Total capacity: ~$(df -h "$MOUNT_POINT" | tail -1 | awk '{print $2}')"
echo "  • Read/write operations are balanced across both disks"
echo ""
echo -e "${YELLOW}Important Reminders:${NC}"
echo "  • RAID 0 has NO redundancy - if one disk fails, all data is lost"
echo "  • Keep regular backups of important data"
echo "  • Monitor disk health regularly with: smartctl -a $DISK1"
echo ""
