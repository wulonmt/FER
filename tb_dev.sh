DIR = $1
MACHINE = $2
ENV = $3
DESCRIPTIOIN = $4
NAME = "${MACHINE}_$ENV"
echo "tensorboard dev upload directory:$DIR"
echo "name: $NAME"
tensorboard dev upload --logdir $DIR --name $NAME --description $DESCRIPTION

