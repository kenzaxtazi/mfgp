#!/bin/bash
#BATCH -p short-serial
#SBATCH --job-name regional_analysis
#SBATCH -o value_multiloc_test-%j.o
#SBATCH -e value_multiloc_test-%j.e
#SBATCH --open-mode=truncate
#SBATCH -t 12:00:00
#SBATCH --mem=8000

python -u value_multiloc_test.py
