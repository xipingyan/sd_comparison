# stable_diffusion_compare
Comparing some version's SD performance and accuracy.

# Python VENV
```
python3 -m venv python-env
source python-env/bin/activate
pip install update
pip install -r requirements.txt
```

# Prepare model
```
mkdir models && cd models
copy "clip-vit-base-patch16, stable-diffusion-v1-4" to here
```

# Run comparison

```
./run.sh
```