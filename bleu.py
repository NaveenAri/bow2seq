import sys
import itertools
from tqdm import tqdm
import nltk

def get_seqs(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		seqs = [line.strip().split for line in lines]
	return seqs
refs = get_seqs(sys.argv[1])
hyps = get_seqs(sys.argv[2])
print '%d refs, %d hyps'%(len(refs), len(hyps))
refs = refs[:len(hyps)]

BLEUscores = []
for ref, hyp in tqdm(itertools.izip(refs,hyps)):
	BLEUscores.append(nltk.translate.bleu_score.sentence_bleu([ref], hyp))

print 'BLEUscore: %f averaged over %d '%(sum(BLEUscore)/float(len(BLEUscores)), len(hyps))