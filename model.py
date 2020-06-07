import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss

try:
    import apex
    from apex import amp
except ModuleNotFoundError:
    apex = None

def to_parallel(args, model):
    if args.n_gpu != 1:
        model = torch.nn.DataParallel(model)
    return model

def save_model(output_dir, model):
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), f"{output_dir}/pytorch_model.bin")

def to_fp16(args, model, optimizer=None):
    if args.fp16 and not args.no_cuda:
        if apex is None:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if optimizer is None:
            model = apex.amp.initialize(model, opt_level=args.fp16_opt_level)
        else:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    if optimizer is None:
        return model
    return model, optimizer

class CNN(nn.Module):
    def __init__(self, args, num_labels, emb_freeze=False, class_weight=None):
        super().__init__()
        self.num_labels = num_labels
        self.emb_freeze = emb_freeze
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight)

        self.char_embedding = nn.Embedding(args.max_chars+2, args.dim)

        self.covns = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=args.filter_num,
                                              kernel_size=(k, self.char_embedding.embedding_dim),
                                              padding=0) for k in args.filters])

        self.dropout = nn.Dropout(args.dropout_ratio)
        self.tanh = nn.Tanh()

        filter_dim = args.filter_num * len(args.filters)
        self.linear = nn.Linear(filter_dim, num_labels)

        if args.fp16:
            self.sigmoid = amp.register_float_function(torch, 'sigmoid')

    def forward(self, inputs, labels=None):
        if self.emb_freeze:
            with torch.no_grad():
                x = self.char_embedding(inputs)
        else:
            x = self.char_embedding(inputs)

        x = x.unsqueeze(1)
        x = [self.tanh(conv(x)).squeeze(3) for conv in self.covns]

        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]
        sentence_features = torch.cat(x, dim=1)

        x = self.dropout(sentence_features)
        logits = self.linear(x)

        logits = torch.sigmoid(logits)

        outputs = (logits,)

        if labels is not None:
            if self.class_weight is not None:
                loss_fct = BCELoss(weight=self.class_weight.cuda())

            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs
