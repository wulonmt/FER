#!/bin/sh

#chmod +x your_script.sh
DIR=$1
MACHINE="IRIS"
ENV=$(echo $DIR | cut -d"/" -f 1)
DESCRIPTION=$(echo $DIR | cut -d"/" -f 2)
NAME="${MACHINE}_Fed_${ENV}"
echo "Tensorboard dev upload directory:$DIR"
echo "This machine: $MACHINE"
echo "Name: $NAME"
echo "Description: $DESCRIPTION"
tensorboard dev upload --logdir $DIR --name $NAME --description $DESCRIPTION

