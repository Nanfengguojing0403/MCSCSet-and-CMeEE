from data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
from GlobalPointer import GlobalPointer, MetricsCalculator
from tqdm import tqdm
from torchcrf import CRF


from torch.utils.tensorboard import SummaryWriter

bert_model_path = 'RoBERTa'  #RoBert_large 路径
train_cme_path = 'datasets/MCSCSet_train.json'  #CMeEE 训练集
eval_cme_path = 'datasets/MCSCSet_valid.json'  #CMeEE 测试集
device = torch.device("cuda:0")
BATCH_SIZE = 8

ENT_CLS_NUM = 9

# tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

# train_data and val_data
ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train, batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=0)
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl, batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=0)

# GP MODEL with CRF
encoder = BertModel.from_pretrained(bert_model_path)
global_pointer = GlobalPointer(encoder, ENT_CLS_NUM, 64).to(device)
crf = CRF(ENT_CLS_NUM, batch_first=True).to(device)

class GPModelWithCRF(torch.nn.Module):
    def __init__(self, global_pointer, crf):
        super(GPModelWithCRF, self).__init__()
        self.global_pointer = global_pointer
        self.crf = crf

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        logits = self.global_pointer(input_ids, attention_mask, token_type_ids)
        batch_size, num_entity_types, seq_len, some_other_dim = logits.shape
        logits = logits.view(batch_size, seq_len, -1)
        if labels is not None:
            # CRF loss calculation
            loss = -self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            # CRF decoding
            prediction = self.crf.decode(logits, mask=attention_mask.byte())
            return prediction

model = GPModelWithCRF(global_pointer, crf).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

writer = SummaryWriter('log_train')
Total_train_step = 0
metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0

for eo in range(10):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
        
        model.train()
        loss = model(input_ids, attention_mask, segment_ids, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logits = global_pointer(input_ids, attention_mask, segment_ids)
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss += loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        print("train_loss:", avg_loss, "\t train_f1:", avg_f1)
        writer.add_scalar('LOSS', avg_loss, Total_train_step)

    with torch.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
            
            prediction = model(input_ids, attention_mask, segment_ids)
            logits = global_pointer(input_ids, attention_mask, segment_ids)
            
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r

        avg_f1 = total_f1_ / (len(ner_loader_evl))
        avg_precision = total_precision_ / (len(ner_loader_evl))
        avg_recall = total_recall_ / (len(ner_loader_evl))
        print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1, avg_precision, avg_recall))
        writer.add_scalar('F1', avg_f1, Total_train_step)
        writer.add_scalar('Precision', avg_precision, Total_train_step)
        writer.add_scalar('Recall', avg_recall, Total_train_step)
        Total_train_step = Total_train_step + 1

        if avg_f1 > max_f:
            torch.save(model.state_dict(), './outputs/ent_model.pth')
            max_f = avg_f1

writer.close()