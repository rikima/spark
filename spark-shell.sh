#!/bin/sh
cur=$(dirname $0)

jars=""
for jar in $(ls $cur/extlib/*.jar) ; do
    jars="$jar,$jars"
done
echo $jars
$cur/bin/spark-shell --jars $jars
