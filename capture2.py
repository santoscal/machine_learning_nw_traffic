import csv
from scapy.all import *
import sys
import time

# declare sniffing_process as a global
sniffing_process = None

# Initialize packet counter variable
packet_count = 0

# Define the loading animation
def loading_animation():
    chars = "/—\|" # The characters used for the animation
    index = 0
    while True:
        # Print the next character in the animation sequence
        sys.stdout.write('\r' + "Capturing... " + chars[index % len(chars)])
        sys.stdout.flush()
        index += 1
        time.sleep(0.1)

# Start the loading animation in a separate thread
import threading
loading_thread = threading.Thread(target=loading_animation)
loading_thread.daemon = True
loading_thread.start()


# print("Capturing...")
# Function to process packets and write to CSV
def process_packet(packet):
    global sniffing_process, packet_count

    src_ip = None
    dst_ip = None
    src_port = None
    dst_port = None
    protocol = None
    timestamp = None

    if IP in packet:
        # Extract packet information
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[IP].sport
        dst_port = packet[IP].dport
        protocol = packet[IP].proto
        timestamp = packet.time

    # Write packet information to CSV file
    with open('packets.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([src_ip, dst_ip, src_port, dst_port, protocol, timestamp])

        # Increment packet counter
        packet_count += 1

        # Stop capturing after 100 packets
        if packet_count >= 100 and sniffing_process is not None:
            sniffing_process.close()

# Write header row to CSV file
with open('packets.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["src_ip", "dst_ip" , "src_port", "dst_port", "protocol", "timestamp"])

# Sniff packets and process them
try:
    sniffing_process = sniff(prn=process_packet, store=False)
except Exception as e:
    print("Error: ")
finally:
    if sniffing_process is not None:
        sniffing_process.close()

# Print the total number of packets captured
print("Total packets captured:", packet_count)
