OUT_DIR=results/human
gunicorn 'src.human.main:app' -w 1 --bind 0.0.0.0:8082 -k gevent --threads 8 

######### FOR LOGGING TO PILOT_LOG #########
PILOT_LOG=${OUT_DIR}/human.log
# gunicorn 'src.human.main:app' -w 1 --bind 0.0.0.0:8082 -k gevent --threads 8 >> ${PILOT_LOG}