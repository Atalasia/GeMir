import torch
import math
from collections import OrderedDict
import argparse
from datetime import datetime

from autoencoder.ae_model import AE
from predictor.model import GeMir


def get_pos_enc():

    d_model = 1
    seq_len = 300000

    position = torch.arange(seq_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(300000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    out_dict = dict()

    for idx, i in enumerate(range(0, seq_len, 1000)):
        out_dict[idx] = pe[idx:idx + 1000]

    return out_dict


def conversion(seq: str):

    seq = seq.upper()
    result = []

    for base in seq:
        if base == "A":
            val = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            result.append(val)
        elif base == "C":
            val = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
            result.append(val)
        elif base == "G":
            val = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
            result.append(val)
        elif base == "U" or base == "T":
            val = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            result.append(val)
        elif base == "0":
            val = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            result.append(val)
        else:
            val = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            result.append(val)

    result = torch.cat(result, dim=0)

    return result

def pad_sequences(seq, target_block_len):

    pad_len = target_block_len - (len(seq) % target_block_len)
    if pad_len != target_block_len:
        return seq + "0" * pad_len
    else:
        return seq

def split_and_convert_sequences(seq):

    seqs = [seq[i:i + 10] for i in range(0, len(seq), 10)]
    res = []

    for seq in seqs:
        res.append(conversion(seq))

    return res


def encode_sequences(mirna_dict, genes_dict, device):

    weights_fp = "autoencoder/weight/seq_enc.pt"

    model = AE()
    model.load_state_dict(torch.load(weights_fp))
    model.to(device)

    enc_mirnas = OrderedDict()

    for mirna_id, mirna in mirna_dict.items():
        mirna = pad_sequences(mirna, 30)
        mirna_vecs = split_and_convert_sequences(mirna)
        mirna_vecs = torch.stack(mirna_vecs, dim=0).to(device)

        enc_val = model(mirna_vecs)
        enc_mirna = torch.cat([enc_val[0],enc_val[1],enc_val[2]], dim=0)
        enc_mirnas[mirna_id] = enc_mirna

    enc_genes = OrderedDict() # list of gene blocks per gene

    for gene_id, gene in genes_dict.items():
        buffer = []
        encoded_blocks = [] ## list of size 1,000 encoded blocks
        gene = pad_sequences(gene, 10000)
        gene_vecs = split_and_convert_sequences(gene) ## list of (4 x 10) tensors

        for idx in range(0, len(gene_vecs), 32):
            final_idx = min(idx + 32, len(gene_vecs))
            batch_vecs = torch.stack(gene_vecs[idx: final_idx], dim=0).to(device)
            enc_batch = model(batch_vecs)

            for enc in enc_batch:
                buffer.append(enc)
                if len(buffer) == 1000:
                    enc_block = torch.cat(buffer, dim=0)
                    encoded_blocks.append(enc_block)
                    buffer = []

        enc_genes[gene_id] = encoded_blocks

    return enc_mirnas, enc_genes


def parse_fasta_file(fp):
    infile = open(fp, "rt")

    parsed_seqs = OrderedDict()
    this_id = None
    curr_seq = []

    for line in infile.readlines():
        line = line.strip()

        if line.startswith(">"):
            if this_id is not None:
                parsed_seqs[this_id] = ''.join(curr_seq)
            this_id = line[1:]
            curr_seq = []
        else:
            curr_seq.append(line)

    parsed_seqs[this_id] = ''.join(curr_seq)

    return parsed_seqs


def format_blocks(gene_block, pe):

    return gene_block


def predict_binding(mirnas, genes, device):

    model = GeMir()
    model.eval()
    model.to(device)

    pe = get_pos_enc()

    result_dict = OrderedDict()

    for i in range(10):
        weights_fp = "predictor/weight/gemir_%s.pt" % i
        model.load_state_dict(torch.load(weights_fp))

        for mirna_id, mirna_b, gene_id, gene_bs in zip(mirnas.items(), genes.items()):

            gene_b = [ format_blocks(gene_b, pe) for gene_b in gene_bs ]
            gene_b = torch.stack(gene_b, dim=0)
            b_count, _, _ = gene_b.shape

            mirna_b = mirna_b.to(device, dtype=torch.float)
            mirna_b = mirna_b.unsqueeze(0).repeat(b_count, 1, 1)

            output = model(mirna_b, gene_b)

            pred = torch.where(output < 0, 0.0, output)
            pred = torch.where(pred > 1.0, 1.0, pred)
            pred_list = [p.item() for p in pred]

            if (mirna_id, gene_id) in result_dict:
                result_dict[(mirna_id, gene_id)].append(pred_list)
            else:
                result_dict[(mirna_id, gene_id)] = list()
                result_dict[(mirna_id, gene_id)].append(pred_list)

    return result_dict


def print_result(result_dict, outfp):

    outfile = open(outfp, "wt")
    outfile.write("miRNA\tgene\tconsensus_blocks\tbinds\n")

    for (mirna, gene), vals in result_dict.items():
        row_count = len(vals[0])
        bindability = list()

        for i in range(row_count):
            consensus = 0
            for val in vals:
                if val[i] >= 0.5:
                    consensus += 1

            if consensus > 5:
                bindability.append(True)
            else:
                bindability.append(False)

        cblocks = ",".join([str(j) for j in bindability])

        if any(bindability):
            res = "True"
        else:
            res = "False"

        outfile.write("%s\t%s\t%s\t%s\n" % (mirna, gene, cblocks, res))

    outfile.flush()
    outfile.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mirna_file', dest='MIRNA_FILE', type=str,
                        help="miRNA fasta file (default: data/mirna.fa)")
    parser.add_argument('-g', '--gene_file', dest='GENE_FILE', type=str,
                        help="gene fasta file (default: data/gene.fa)")

    parser.add_argument('-o','--output_file', dest='OUTPUT_FILE', type=str,
                        help="output file to be saved in 'predict' mode (default: results.csv)")

    args = parser.parse_args()

    return args


def main(configs):

    mirna_fp = configs.MIRNA_FILE
    gene_fp = configs.GENE_FILE
    out_fp = configs.OUTPUT_FILE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mirna_fasta_dict = parse_fasta_file(mirna_fp)
    gene_fasta_dict = parse_fasta_file(gene_fp)


    start_time = datetime.now()
    print("\nStarting prediction at {}".format(start_time.strftime('%Y-%m-%d - %H:%M:%S')))

    enc_mirnas, enc_genes = encode_sequences(mirna_fasta_dict, gene_fasta_dict, device)
    result_dict = predict_binding(enc_mirnas, enc_genes, device)
    print_result(result_dict, out_fp)

    finish_time = datetime.now()
    print("\nFinished prediction at {} (took {} seconds)\n".
          format(finish_time.now().strftime('%Y-%m-%d - %H:%M:%S'), (finish_time - start_time)))


configs = parse_arguments()
main(configs)
