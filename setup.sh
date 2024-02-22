conda create -n aft python=3.9
conda activate aft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib
pip install seaborn
pip install wandb
pip install transformers[torch] datasets evaluate scikit-learn
pip install fire
pip install timm
pip install ftfy regex tqdm numba
pip install git+https://github.com/openai/CLIP.git
