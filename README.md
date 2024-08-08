# GeMir


Holistic prediction of miRNA and gene.

## Requirements:
* pytorch
* pandas
* numpy

## Usage:
```
python predict.py -m <mirna fasta file> -g <gene fasta file> -o <outfile>
```
Write a pair of miRNA, gene sequences in each fasta file to predict if the pair binds or not. 
