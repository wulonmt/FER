#!/bin/sh
DIR=$1
MACHINE=$2
ENV=$3
DESCRIPTION=$4
NAME="${MACHINE}_${ENV}"
echo "tensorboard dev upload directory:$DIR"
echo "name: $NAME"
echo "description: $DESCRIPTION"
tensorboard dev upload --logdir $DIR --name $NAME --description $DESCRIPTION

