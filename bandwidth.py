import speedtest

st = speedtest.Speedtest()
download_speed = st.download() / 1000000
upload_speed = st.upload() / 1000000

print("Download speed: {:.2f} Mbps".format(download_speed))
print("Upload speed: {:.2f} Mbps".format(upload_speed))