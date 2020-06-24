start=0
end=200000
image_dir="./dataset/images"
metadata_dir="./dataset/metadata"

for i in $(seq -f '%05g' $start $end); do
    wget https://aft-vbi-pds.s3.amazonaws.com/metadata/$i.json -P $metadata_dir &
    wget https://aft-vbi-pds.s3.amazonaws.com/bin-images/$i.jpg -P $image_dir
done
