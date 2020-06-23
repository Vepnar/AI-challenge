# Download from 0 to 200 to get all samples with 0 items 
# Download from 201 to 400 to get all samples with 1 item
# Download from 401 to 600 to get all samples with 2 items
# Download form 601 to 800 to get all samples with 3 items
# 800 to 100 for 4 items
# 1001 to 1200 for 5 items


start=201
end=301
image_dir="./dataset/images"
metadata_dir="./dataset/metadata"

for i in $(seq $start $end); do
    wget https://aft-vbi-pds.s3.amazonaws.com/metadata/$i.json -P $metadata_dir &
    wget https://aft-vbi-pds.s3.amazonaws.com/bin-images/$i.jpg -P $image_dir
done
