import os
import re
import bz2
import json
import gzip
import glob
import gensim
import zipfile

from multiprocessing import Pool
import multiprocessing as multi

from collections import Counter, defaultdict
from preprocess import load_wiki_dump, load_cirrus_dump

import torch
from torch.utils.data import Dataset

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def load_oneliner_json(file_path, debug=False):
    data = []
    with open(file_path, "r") as f, \
         Pool(multi.cpu_count()) as p:
        for d in p.imap(json.loads, f):
            data.append(d)
            if debug and len(data) >= 10000:
                break
    return data

def calc_f1(tp, tpfp, tpfn):
    if not tp:
        return 0.0, 0.0, 0.0
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = 2*precision*recall/(precision+recall)
    return precision*100, recall*100, f1*100

def average(array):
    if not len(array):
        return 0.0
    return sum(array) / len(array)

def average_score(score):
    recalls, precisions, f1s = zip(*score.values())
    return average(recalls), average(precisions), average(f1s)

class ShinraDataset(Dataset):
    def __init__(self, args, part="train", labels=None, chars=None, data=None):
        self.data = data
        if data is None:
            file_path = os.path.join(args.dataset, args.lang, f"{part}.json")
            self.data = load_oneliner_json(file_path, args.debug)
        self.seq_len = args.seq_len
        self.max_chars = args.max_chars
        self.lang = args.lang

        self.chars = chars
        if chars is None:
            self.chars = self._char_count()
        self.pad_id = len(self.chars)
        self.unk_id = len(self.chars) + 1
        self.char2id = {c:idx for idx, c in enumerate(self.chars)}

        self.labels = labels
        if labels is None:
            ene_path = os.path.join(args.dataset, args.lang, "ene_set.json")
            self.labels = load_json(ene_path)
        self.label2idx = {l:idx for idx,l in enumerate(self.labels)}
        self.idx2label = {v:k for k,v in self.label2idx.items()}
        self.num_labels = len(self.labels)

        self.answers = {d["pageid"]:d.get("label") for d in self.data}
        self.pageid2title = {d["pageid"]:d["title"] for d in self.data}

    def _padding(self, array, pad, length):
        return array + [pad] * (length - len(array))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        text = d["text"][:self.seq_len]
        char_ids = [self.char2id.get(c, self.unk_id) for c in text]
        char_ids = self._padding(char_ids, self.pad_id, self.seq_len)
        label_ids = [float(l in d["label"]) for l in self.labels]
        return {
            "inputs":torch.LongTensor(char_ids),
            "label":torch.FloatTensor(label_ids),
            "pageid":str(d["pageid"])
        }

    def _decode(self, results):
        decoded = {}
        for pageid, result in results:
            d = {"pageid":pageid, "title":self.pageid2title[pageid], "lang":self.lang, "ENEs":[]}
            flags = [(idx, prob) for idx, prob in enumerate(result) if prob >= 0.5]
            if len(flags) == 0:
                max_prob = max(result)
                idx = result.index(max_prob)
                d["ENEs"].append({"ENE_id":self.idx2label[idx], "prob":max_prob})
            else:
                for idx, prob in flags:
                    d["ENEs"].append({"ENE_id":self.idx2label[idx], "prob":prob})
            decoded[pageid] = d
        return decoded

    def _char_count(self):
        stack = ""
        for d in self.data:
            stack += d["text"][:self.seq_len]

        cnt = Counter(stack)
        chars = [char for char, _ in cnt.most_common()]
        return chars[:self.max_chars]

    def get_class_weight(self):
        labels = []
        for pageid, label in self.answers.items():
            labels += label
        cnt = Counter(labels)
        total = [cnt[label] for label in self.labels]
        max_num = max(total)
        return [max_num/num if num != 0 else 1 for num in total]

    def evaluate(self, results):
        results = self._decode(results)

        tp, tpfp, tpfn = defaultdict(int), defaultdict(int), defaultdict(int)
        for pageid, answer in self.answers.items():
            result_ene_ids = [ene["ENE_id"] for ene in results[pageid]["ENEs"]]
            for ene in set(answer) & set(result_ene_ids):
                tp[ene] += 1
            for ene in set(result_ene_ids):
                tpfp[ene] += 1
            for ene in set(answer):
                tpfn[ene] += 1

        score = {}
        for ene in self.labels:
            score[ene] = calc_f1(tp[ene], tpfp[ene], tpfn[ene])
        score["macro ave"] = average_score(score)
        score["micro ave"] = calc_f1(sum(tp.values()), sum(tpfp.values()), sum(tpfn.values()))

        return score, results

def load_shinra_data(args):
    train_dataset = ShinraDataset(args, part="train")
    dev_dataset = ShinraDataset(args, part="dev", labels=train_dataset.labels, chars=train_dataset.chars)
    return train_dataset, dev_dataset

def save_result(file_path, data):
    with Pool(multi.cpu_count()) as p:
        dumps = p.map(json.dumps, data.values())
    with open(file_path, "w") as f:
        f.write("\n".join(dumps))

class  ShinraTargetDataset(ShinraDataset):
    def __init__(self, args, labels, chars):
        data = self._load_corpus(args.input_dir, args.lang, args.num_workers)
        super().__init__(args, data=data, labels=labels, chars=chars)

    def __getitem__(self, i):
        d = self.data[i]
        text = d["text"][:self.seq_len]
        char_ids = [self.char2id.get(c, self.unk_id) for c in text]
        char_ids = self._padding(char_ids, self.pad_id, self.seq_len)
        return {
            "inputs":torch.LongTensor(char_ids),
            "pageid":str(d["pageid"])
        }

    def _load_corpus(self, input_dir, target_lang, num_workers):
        dump = None
        for zip_file_path in glob.glob(f"{input_dir}/*.zip"):
            with zipfile.ZipFile(zip_file_path) as zf:
                organized = defaultdict(dict)
                for file_path in zf.namelist():
                    *_, lang, name = file_path.split(os.sep)

                    if lang != target_lang:
                        continue

                    if re.match(".+?cirrussearch-content.json.gz", file_path) is not None:
                        with zf.open(file_path) as f:
                            with gzip.open(f, mode='rt') as gzf:
                                dump = load_cirrus_dump(gzf, num_workers=num_workers)
                    elif re.match(".+?wiki-20190120-pages-articles.xml.bz2", file_path) is not None:
                        with zf.open(file_path) as f:
                            with bz2.open(f, mode='rt') as bz2f:
                                dump = load_wiki_dump(bz2f.read())
        if dump is None:
            raise FileNotFoundError()
        return list(dump.values())

    def evaluate(self, results):
        results = self._decode(results)
        return None, results
