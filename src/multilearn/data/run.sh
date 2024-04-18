export PYTHONPATH="../MAST-ML/":$PYTHONPATH
python3 clean.py
python3 split.py
rm -rf Elemental*
