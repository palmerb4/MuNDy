# Instructions for installing JUBE

```bash
> module load python/3.10.10
> mkdir -p ~/envs
> python3 -m venv --system-site-packages ~/envs/jube
> source ~/envs/jubs/bin/activate
> pip3 install http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=latest \
  --user
> jube --version
```
