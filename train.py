import os
import json
import tqdm
import argparse

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, Adagrad
from torch.optim.lr_scheduler import LambdaLR

from data_utils import load_shinra_data, save_json
from model import CNN, to_parallel, to_fp16, save_model

try:
    import apex
    from apex import amp
except ModuleNotFoundError:
    apex = None

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max_chars", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=300)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")

    parser.add_argument("--filter_num", type=int, default=500)
    parser.add_argument('--filters', nargs='*', default=[3,5,7,9,11])

    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size",type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--emb_freeze", action="store_true")

    parser.add_argument("--optim", choices=["adam", "adagrad"], default="adam")
    parser.add_argument("--lr_dec_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--adam_B1", type=float, default=0.9)
    parser.add_argument("--adam_B2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-6)

    parser.add_argument("--dropout_ratio", type=float, default=0.5)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def set_seed(args):
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def print_score(score, desc="TRAIN"):
    print(f"|{desc:<7}|{score[0]:>6.2f}|{score[1]:>6.2f}|{score[2]:>6.2f}|")

def train(args, dataset, dev_dataset, model, do_train=True):
    model.train()
    dataloader = DataLoader(dataset, shuffle=args.shuffle, batch_size=args.batch_size, num_workers = args.num_workers)
    args.total_steps = len(dataloader) * args.epoch // args.gradient_accumulation_steps

    lr_lambda = lambda epoch: 0.9 ** (epoch)
    if args.optim == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr,
                          betas=(args.adam_B1, args.adam_B2), weight_decay=args.weight_decay, eps=args.adam_eps)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        optimizer = Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.fp16:
        model, optimizer = to_fp16(args, model, optimizer)

    model = to_parallel(args, model)

    steps = 0
    best_score = defaultdict(lambda:[0, 0, 0])
    best_epoch = None

    for epoch in range(1,args.epoch+1):
        if not do_train:
            break

        outputs = []
        tr_loss = []
        if epoch >= args.lr_dec_epoch:
            scheduler.step()

        for batch in tqdm.tqdm(dataloader, desc=f"TRAIN {epoch}"):
            loss, logit, *_ = model(inputs=batch["inputs"].to(args.device),
                          labels=batch["label"].to(args.device))

            outputs += list(zip(batch["pageid"], logit.cpu().tolist()))

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss.append(loss.item())

            steps += 1
            if steps % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
        print(f"|LOSS|{sum(tr_loss)/len(tr_loss)}|LR|{scheduler.get_lr()}|")

        score, _ = dataset.evaluate(outputs)
        print_score(score["micro ave"])

        dev_score, _ = eval(args, dev_dataset, model)
        print_score(dev_score["micro ave"], desc="DEV")
        if dev_score["micro ave"][-1] > best_score["micro ave"][-1]:
            best_epoch = epoch
            best_score = dev_score
            best_model_param = model.state_dict()

        model.freeze = True

    if best_score is None:
        best_score, _ = eval(args, dev_dataset, model)
    else:
        model.load_state_dict(best_model_param)

    print_score(best_score["micro ave"], desc="BEST DEV")

    score, _ = eval(args, dataset, model)
    print_score(score["micro ave"], desc="BEST(DEV) TRAIN")

    return model, score, best_score, best_epoch

def eval(args, dataset, model):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers = args.num_workers)

    outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="EVAL"):
            logit, *_ = model(inputs=batch["inputs"].to(args.device))

            outputs += list(zip(batch["pageid"], logit.cpu().tolist()))

    return dataset.evaluate(outputs)

def main():
    args = load_arg()
    print(f"Run:{args.lang}")

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    train_dataset, dev_dataset = load_shinra_data(args)
    model = CNN(args, train_dataset.num_labels, \
                emb_freeze=args.emb_freeze, \
                class_weight=train_dataset.get_class_weight())

    model.to(args.device)

    scores = {}
    model, scores["train"], scores["dev"], best_epoch = train(args, train_dataset, dev_dataset, model)

    model.to("cpu")

    if args.output_dir is not None:
        output_dir = f"{args.output_dir}/{args.lang}"
        os.makedirs(output_dir, exist_ok=True)
        save_model(output_dir, model)
        save_json(f"{output_dir}/score.json", scores)

if __name__ == '__main__':
    main()
