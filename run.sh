SCENE=$1
#sh colmap.sh data/$SCENE
python3 train.py --config configs/$SCENE.txt  --no_ndc --spherify --lindisp --expname=$SCENE
python eva.py --config configs/$SCENE.txt  --no_ndc --spherify --lindisp --expname=$SCENE

