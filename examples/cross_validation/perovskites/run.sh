export PYTHONPATH=$(pwd)/../../../src/:$PYTHONPATH
rm -rf outputs
torchrun fit.py
