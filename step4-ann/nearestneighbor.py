from annoy import AnnoyIndex
import numpy as np

with open("vectors.txt", "r") as out:
    files = out.read().splitlines()

f = 1000
t = AnnoyIndex(f, 'angular')
for i, file in enumerate(files):
    v = np.load('/N/project/htrc/data/ACS/2019/krewson/vectors/' + file)[0]
    t.add_item(i, v)
    print('[%d] Added %s to the index...' % (i, file))

print('Building trees...')
t.build(100)  # 100 trees

print('Saving index...')
t.save('/N/project/htrc/data/ACS/2019/krewson/htrc.ann')

print('All done')

# ...

# u = AnnoyIndex(f, 'angular')
# u.load('htrc.ann')
# neighbors = u.get_nns_by_item(0, 10)
#
# for n in neighbors:
#     print(files[n])
