import sys
import itertools
from tqdm import tqdm
import nltk

def get_seqs(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		seqs = [line.lower().strip().split() for line in lines]
	return seqs
refs = get_seqs(sys.argv[1])
hyps = get_seqs(sys.argv[2])
print '%d refs, %d hyps'%(len(refs), len(hyps))
refs = refs[:len(hyps)]

BLEUscores = []
for ref, hyp in tqdm(itertools.izip(refs,hyps)):
	temp = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
	BLEUscores.append(temp)

print 'BLEUscore: %f averaged over %d '%(sum(BLEUscores)/float(len(BLEUscores))*100, len(hyps))