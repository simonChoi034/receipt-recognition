import argparse
import json
import os
import re


def main(args):
    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if re.match(r'X[0-9]+\.txt', f)]
    output_file = input("Output dir: ")

    for file in files:
        try:
            word_list = read_file(file)
            word_class = [{"word": word, "class": 0} for word in word_list]
            filename = file.replace(args.dir + '/', '').replace('.txt', '')

            json_file = {
                "data": word_class
            }

            with open(output_file + '/' + filename + '.json', 'w') as f:
                json.dump(json_file, f, indent=4)

        except:
            pass


def read_file(file):
    word_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.split(',', 8)
            words = line[-1].replace('\n', '')
            words = words.split()
            word_list.extend(words)

    return word_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('-d', '--dir', help='Directory of dataset', required=True)
    args = parser.parse_args()

    main(args)
