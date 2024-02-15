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

To run JUBE in order, you need to first create/submit the benchmark, and then you can 
go back and analyze it later. This is done by ID. Also, check for hard-coded paths
in `mundy_brownian_strongscaling.yaml` as right now this is coded up for Chris Edelmaier
and his build structure.

## JUBE creation/submission

```bash
> jube run mundy_brownian_strongscaling.yaml
```

Note that this gives you an ID of what run it was.

## JUBE analyze (analyse) results

```bash
> jube analyse mundy_brownian_strongscaling_rome_cpu --id 0
```

Note use the ID that you want to analyze from the first step.

## JUBE results

```bash
> jube result mundy_brownian_strongscaling_rome_cpu --id 0
> jube result mundy_brownian_strongscaling_rome_cpu --id 0 --style csv
```

The second command dumps a CSV form to screen that is nice for reading into python/pandas.
