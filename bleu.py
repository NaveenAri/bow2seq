import sys
import itertools
from collections import defaultdict
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

len_BLEUscores = defaultdict(list)
BLEUscores = []
for ref, hyp in tqdm(itertools.izip(refs,hyps)):
	if len(ref)<5 or len(hyp)<5:
		continue
	score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)*100
	BLEUscores.append(score)
	len_BLEUscores[len(ref)].append(score)

print 'BLEUscore: %f averaged over %d '%(sum(BLEUscores)/float(len(BLEUscores)), len(hyps))
lens = sorted(len_BLEUscores.keys())
avgBLEUscores = [sum(len_BLEUscores[l])/float(len(len_BLEUscores[l])) for l in lens]

print lens, avgBLEUscores

import matplotlib.pyplot as plt
plt.plot(lens, avgBLEUscores, '-')
plt.ylabel('BLEU')
plt.xlabel('len')
plt.savefig('BLEU_vs_len.png', bbox_inches='tight')
plt.show()

