source /etc/profile

find "../Dataset/" -type f -exec md5sum {} \; | sort > md5_hashes_temp.txt
sort md5_hashes.txt -o md5_hashes.txt
diff md5_hashes_temp.txt md5_hashes.txt
rm md5_hashes_temp.txt