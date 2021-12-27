import argparse
from utils import sample, ModelStorage, NgramStorage
import ngrams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a pretrained language model')
    parser.add_argument('--lstm_path', default='', type=str, help='Path to a pre-trained LSTM model')
    parser.add_argument('--ngram_path', default='', type=str, help='Path to a pre-trained N-gram model')
    parser.add_argument('--prompt', type=str, default='', help='Text used to condition a model on')
    args = parser.parse_args()
    prompt = args.prompt

    print(f'Prompt:\n"{prompt}"')

    if args.lstm_path:
        net, encoder = ModelStorage.load(args.lstm_path)
        tokens = sample(net, encoder, prompt, steps=100)
    elif args.ngram_path:
        model, encoder = NgramStorage.load(args.ngram_path)
        tokens = ngrams.sample(model, encoder, prompt, 2)
    else:
        raise Exception('Argument error: --lstm_path or --ngram_path argument has to be specified')

    result = ' '.join(tokens)
    print(f'Generated text (including prompt):\n"{result}"')
