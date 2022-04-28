#!/bin/bash
#SBATCH -p short-serial
#SBATCH --job-name regional_analysis
#SBATCH -o value_multiloc_test_alt-%j.o
#SBATCH -e value_multiloc_test_alt-%j.e
#SBATCH --open-mode=truncate
#SBATCH -t 24:00:00
#SBATCH --mem=100G

python -u value_multiloc_test_alt.py
