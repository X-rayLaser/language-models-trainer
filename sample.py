import argparse
from utils import sample, ModelStorage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a pretrained language model')
    parser.add_argument('pretrained_path', type=str, help='Path to a pre-trained model')
    parser.add_argument('--prompt', type=str, default='', help='Text used to condition a model on')
    args = parser.parse_args()
    net, encoder = ModelStorage.load(args.pretrained_path)
    prompt = args.prompt

    print(f'Prompt:\n"{prompt}"')
    tokens = sample(net, encoder, prompt, steps=100)

    result = ' '.join(tokens)
    print(f'Generated text (including prompt):\n"{result}"')
