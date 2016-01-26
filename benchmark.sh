#!/bin/bash

echo "----1. Finding the largest block that fits into the GPU...---"

nelem=100 # number of millions
for i in {1..7}
do
    echo "./gendata ${nelem}m"
    ./gendata ${nelem}m 2>/dev/null | sed 's/^/    /'

    if [ ${PIPESTATUS[0]} -eq 0 ]
    then
        (( nelem *= 2 ))
    else
        echo "    --> Nope, does not fit. Reducing data size by 10%..."
        break
    fi
done

for i in {1..5}
do
    (( nelem = nelem * 90 / 100 ))
    echo "./gendata ${nelem}m"
    ./gendata ${nelem}m 2>/dev/null | sed 's/^/    /'

    if [ ${PIPESTATUS[0]} -eq 0 ]
    then
        break
    else
        echo "    --> Nope, does not fit. Reducing data size by 10%..."
    fi
done
echo "./gendata 'dump.bin' ${nelem}m"
./gendata 'dump.bin' ${nelem}m 2>/dev/null | sed 's/^/    /'

echo ""
echo "----2. Running sorting benchmark...---"
while (( nelem > 1))
do
    echo "./sort 'dump.bin' ${nelem}m"
    ./sort 'dump.bin' ${nelem}m 2>/dev/null | sed 's/^/    /' \
        | tee -a log.txt

    if [ ${PIPESTATUS[0]} -eq 0 ]
    then
        (( nelem /= 2 ))
        echo "" >> log.txt
    else
        echo -n "    --> Insufficient memory for sorting. "
        echo "Reducing data size by 20%..."
        (( nelem = nelem * 80 / 100 ))
        > log.txt
    fi
done

sed -i -e 's/^    //g' log.txt
