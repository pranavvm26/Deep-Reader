"""Dataset reader and process"""

import os
import html
import string
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
import preprocess.image_preprocessing as pp
from sklearn.model_selection import train_test_split


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, destination, name, data_source='words'):
        self.source = source
        self.destination = destination
        self.name = name
        self.dataset = None
        self.data_source = data_source
        self.partitions = ['train', 'valid', 'test']

        os.makedirs(self.destination, exist_ok=True)

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")(self.data_source)

        if not self.dataset:
            self.dataset = dict()

            for y in self.partitions:
                self.dataset[y] = {'dt': [], 'gt': []}

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def preprocess_partitions(self, input_size):
        """Preprocess images and sentences from partitions"""

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[y]['gt'][i])

                if not self.check_text(text):
                    self.dataset[y]['gt'].pop(i)
                    self.dataset[y]['dt'].pop(i)
                    continue

                self.dataset[y]['gt'][i] = text.encode()

            results = []

            print(f"Partition: {y}")
            for imgpath in tqdm(self.dataset[y]['dt'], total=len(self.dataset[y]['dt'])):
                try:
                    results.append(pp.preprocess(imgpath, input_size=input_size))
                except Exception as e:
                    print("\n Exception Error", e, imgpath, "\n")

            self.dataset[y]['dt'] = results

    def _iam(self, data_source):
        """IAM dataset reader"""

        lines = open(os.path.join(self.source, "ascii", f"{data_source}.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()

            if split[1] == "ok":
                gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        dataset = dict()

        lines_train_valid, lines_test = train_test_split(lines, test_size=0.1, random_state=9)
        lines_train, lines_valid = train_test_split(lines_train_valid, test_size=0.1, random_state=9)

        for t, line_partition in zip(self.partitions, [lines_train, lines_valid, lines_test]):
            dataset[t] = {"dt": [], "gt": []}

            for line_ in line_partition:
                if line_[0] != '#':
                    line = line_.split()[0]
                    try:
                        split = line.split("-")
                        folder = f"{split[0]}-{split[1]}"

                        img_file = f"{line}.png"
                        img_path = os.path.join(self.source, f"{data_source}", split[0], folder, img_file)

                        dataset[t]['gt'].append(gt_dict[line])
                        dataset[t]['dt'].append(img_path)
                    except KeyError:
                        pass
                else:
                    pass

        return dataset

    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) > 2 and punc_percent <= 0.1
