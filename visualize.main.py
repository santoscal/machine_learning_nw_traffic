from collections import Counter
import csv
import matplotlib.pyplot as plt

#read the csv file and extract  the source ip addresses.
source_ips = []
with open('packets.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        source_ips.append(row['src_ip'])


#count the frequency of source ip   cle

ip_counts = Counter(source_ips)

#sort the ip addresses based on their count.
sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)


plt.bar(range(len(sorted_ips)), [val[1] for val in sorted_ips], align='center')
plt.xticks(range(len(sorted_ips)), [val[0] for val in sorted_ips])
plt.xlabel('Source IP')
plt.ylabel('Count')
plt.title('Count of Source IP Addresses')
plt.show()



