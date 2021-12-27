from datasets import NltkDataset
from ngrams import NGramModel
from utils import NgramStorage

ds = NltkDataset('brown', categories=None)
train_fragments = ds.get_training_fragments()
val_fragments = ds.get_validation_fragments()
test_fragments = ds.get_test_fragments()

save_dir = 'checkpoints/ngrams/bi_grams'

model, encoder = NGramModel.build_model(ds, save_dir, ngram_order=3, max_size=10000)
params = dict(save_dir=save_dir, smoothing=False)
NgramStorage.save(model, params, encoder, save_dir)
model.close()

# todo: separate training from evaluation and sampling
# todo: faster algorithm for building count tables (do not scan the whole dataset multiple times, batch write)
# todo: implement effective smoothing techniques
