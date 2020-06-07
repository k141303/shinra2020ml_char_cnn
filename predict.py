import os
import argparse

import torch

from data_utils import load_shinra_data, save_json, ShinraTargetDataset, save_result
from model import CNN, to_parallel, to_fp16, save_model
from train import eval

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset")
    parser.add_argument("--pretrained_model", type=str, default="./result")
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max_chars", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--dim", type=int, default=300)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")

    parser.add_argument("--filter_num", type=int, default=500)
    parser.add_argument('--filters', nargs='*', default=[3,5,7,9,11])

    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--eval_batch_size",type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--dropout_ratio", type=float, default=0.0)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main():
    args = load_arg()
    print(f"Run:{args.lang}")

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    train_dataset, dev_dataset = load_shinra_data(args)
    dataset = ShinraTargetDataset(args, labels=train_dataset.labels, chars=train_dataset.chars)

    model = CNN(args, train_dataset.num_labels,
                class_weight=train_dataset.get_class_weight())

    state_dict = torch.load(os.path.join(args.pretrained_model, args.lang, "pytorch_model.bin"))
    model.load_state_dict(state_dict)

    model.to(args.device)

    if args.fp16:
        model = to_fp16(args, model)

    model = to_parallel(args, model)

    _, results = eval(args, dataset, model)

    os.makedirs(args.output_dir, exist_ok=True)
    save_result(f"{args.output_dir}/{args.lang}.json", results)

if __name__ == '__main__':
    main()
