import sys

with open(sys.argv[1], 'r') as f:
	lines = f.readlines()

dev_losses = []
train_losses = []
for i in xrange(len(lines)):
	if 'early_stop' in lines[i] and 'out of' in lines[i]:
		dev_loss = lines[i-1].split()[-1][:-2]
		dev_losses.append(dev_loss)
	if 'loss' in lines[i] and '}}' in lines[i]:
		train_loss = lines[i].split()[-1][:-2]
		train_losses.append(train_loss)
dev_losses = dev_losses[:-1]
train_losses = train_losses[:-1]
print dev_losses
print train_losses

import matplotlib.pyplot as plt
plt.plot(dev_losses)
plt.plot(train_losses)
plt.ylabel('loss')
plt.ylim(ymin=0)
plt.savefig('loss.png')
#plt.show()
