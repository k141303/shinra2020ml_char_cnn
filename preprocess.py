import os
import re
import bz2
import gzip
import tqdm
import glob
import json
import random
import zipfile
import argparse
import traceback

from collections import defaultdict
from multiprocessing import Pool
import multiprocessing as multi
from html.parser import HTMLParser

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str,
        help="The path to the directory containing each zip file, \
              such as minimum_en.zip or trial_en.zip.")
    parser.add_argument("-o", "--output_dir", type=str, default="./dataset")
    parser.add_argument("--num_workers", type=int, default=1,
        help="Number of parallel CPUs during file operation.\
              If this argument is set to -1, the maximum number of cores will be set.")
    parser.add_argument("--target_langs", nargs="*", default=None,
        help="If you want to choose the target language, set this argument.")
    parser.add_argument("--dev", type=float, default=0.1,
        help="If you want to change the percentage of development data, set this argument.")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def load_jsons(d):
    try:
        return json.loads(d)
    except json.decoder.JSONDecodeError:
        return None
    except:
        traceback.print_exc()
        import sys
        sys.exit()

def load_enew_list(file_object, num_workers=1):
    num_workers = multi.cpu_count() if num_workers == -1 else num_workers
    enew_list = {}
    ene_set = set()
    with Pool(num_workers) as p:
        for d in p.imap_unordered(json.loads, file_object):
            ENE_ids = [ene["ENE_id"] for ene in d["ENEs"]]
            enew_list[str(d["pageid"])] = ENE_ids
            ene_set |= set(ENE_ids)
    return enew_list, sorted(ene_set)

def load_cirrus_dump(file_object, num_workers=1, pageids=None, debug=False):
    texts = {}
    num_workers = multi.cpu_count() if num_workers == -1 else num_workers
    with tqdm.tqdm(desc="CirrusDump Loading") as process_bar, \
                Pool(multi.cpu_count()) as p:
        for line in p.imap(load_jsons, file_object):
            if line is None:
                continue
            if line.get("index", {}).get("_id"):
                pageid = str(line["index"]["_id"])
            elif pageids is None or pageid in pageids:
                try:
                    texts[pageid] = {"pageid":pageid,
                                     "title":line["title"],
                                     "text":line["text"]}
                    process_bar.update()
                except KeyError:
                    print(f"KeyError:{pageid}")
            if debug and len(texts) >= 100:
                break
    return texts

class MyXMLParser(HTMLParser):
    def __init__(self, pageids=None):
        super().__init__()
        self.pageids = pageids

        self.in_flag = defaultdict(int)
        self.texts = defaultdict(list)
        self.text = defaultdict(str)

        self.pages = {}

    def handle_starttag(self, tag, attrs):
        self.in_flag[tag] += 1

    def handle_endtag(self, tag):
        if self.in_flag[tag] == 0:
            return

        self.in_flag[tag] -= 1
        if self.in_flag[tag] != 0:
            return

        if tag == "page":
            try:
                d = {"pageid":self.texts["id"][0],
                     "title":self.texts["title"][0],
                     "text":self.texts["text"][0]}
                if self.pageids is None or d["pageid"] in pageids:
                    self.pages[d["pageid"]] = d
            except IndexError:
                import traceback
                traceback.print_exc()
                pass
            self.in_flag = defaultdict(int)
            self.texts = defaultdict(list)
            self.text = defaultdict(str)
        else:
            if self.text[tag]:
                self.texts[tag].append(self.text[tag])
                self.text[tag] = ""

    def handle_data(self, data):
        for tag, flag in self.in_flag.items():
            if flag == 0:
                continue
            self.text[tag] += data

def load_wiki_dump(text, pageids=None):
    parser=MyXMLParser(pageids)
    parser.feed(text)
    return parser.pages

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def json_dumps(d):
    return json.dumps(d, ensure_ascii=False)

def save_oneline_json(file_path, data):
    header = True
    with Pool(multi.cpu_count()) as p, \
         open(file_path, "w") as f, \
         tqdm.tqdm(total=len(data), desc="Save") as process_bar:
        for line in p.imap_unordered(json_dumps, data):
            if not header:
                line = "\n" + line
            f.write(line)
            header = False
            process_bar.update()

if __name__ == '__main__':
    args = load_arg()

    file_desc_table = [["lang", "ENE", "Cirrus_dump", "Train data", "Dev data"],
                       ["---", "---", "---", "---", "---"]]
    for zip_file_path in tqdm.tqdm(glob.glob(f"{args.input_dir}/*.zip"), desc="Load"):
        with zipfile.ZipFile(zip_file_path) as zf:
            organized = defaultdict(dict)
            for file_path in zf.namelist():
                *_, lang, name = file_path.split(os.sep)
                if re.match(".+?_ENEW_LIST.json", file_path) is not None:
                    organized[lang]["ENE"] = file_path
                    continue
                elif re.match(".+?cirrussearch-content.json.gz", file_path) is not None:
                    organized[lang]["cirrus"] = file_path
                    continue
                organized[lang]["wikidump"] = file_path

            iter = tqdm.tqdm(organized.items())
            for lang, file_paths in iter:
                if args.target_langs is not None \
                    and lang not in args.target_langs:
                    continue

                iter.set_description(lang)

                with zf.open(file_paths["ENE"]) as f:
                    enew_list, ene_set = load_enew_list(f, args.num_workers)
                pageids = set(enew_list.keys())

                if file_paths.get("cirrus") is not None:
                    with zf.open(file_paths["cirrus"]) as f:
                        with gzip.open(f, mode='rt') as gzf:
                            dump = load_cirrus_dump(gzf, num_workers=args.num_workers,
                                                          pageids=pageids,
                                                          debug=args.debug)
                else:
                    with zf.open(file_paths["wikidump"]) as f:
                        with bz2.open(f, mode='rt') as bz2f:
                            dump = load_wiki_dump(bz2f.read(), pageids=pageids)

                for pageid in dump.keys():
                    dump[pageid]["label"] = enew_list[pageid]

                _, data = map(list, zip(*sorted(dump.items())))

                random.seed(args.seed)
                random.shuffle(data)

                dev_part = int(len(data) * args.dev)
                dev_data = data[:dev_part]
                train_data = data[dev_part:]

                output_dir = os.path.join(args.output_dir, lang)
                os.makedirs(output_dir, exist_ok=True)

                save_json(os.path.join(output_dir, "ene_set.json"), ene_set)
                save_oneline_json(os.path.join(output_dir, "train.json"), train_data)
                save_oneline_json(os.path.join(output_dir, "dev.json"), dev_data)

                file_desc_table.append([lang, len(pageids), len(dump),
                                        len(train_data), len(dev_data)])

            iter.close()

    for row in file_desc_table:
        row = map(lambda x:f"{x:^10}", row)
        print(f"|{'|'.join(row)}|")
