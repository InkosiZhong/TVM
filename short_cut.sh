LOG_HOME=out/results
DATASET=night_street
python -u query/examples/$DATASET/car_tiled.py > $LOG_HOME/$DATASET.tiled.txt
python -u query/examples/$DATASET/car_untiled.py > $LOG_HOME/$DATASET.untiled.txt

DATASET=amsterdam
python -u query/examples/$DATASET/car_tiled.py > $LOG_HOME/$DATASET.tiled.txt
python -u query/examples/$DATASET/car_untiled.py > $LOG_HOME/$DATASET.untiled.txt

DATASET=canal
python -u query/examples/$DATASET/boat_tiled.py > $LOG_HOME/$DATASET.tiled.txt
python -u query/examples/$DATASET/boat_untiled.py > $LOG_HOME/$DATASET.untiled.txt

DATASET=archie
python -u query/examples/$DATASET/car_tiled.py > $LOG_HOME/$DATASET.tiled.txt
python -u query/examples/$DATASET/car_untiled.py > $LOG_HOME/$DATASET.untiled.txt