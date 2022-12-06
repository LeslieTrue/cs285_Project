
from src.irl_maxent.rewardTransformer import RewardTransformer
import torch.nn as nn
import torch.optim as optim
import torch as th

class TransformerTrainer(nn.Module):
    def __init__(self, vocab_size, d_model, input_length, output_length, n_layers, d_filter, dropout=0, learning_rate=1e-3, performer=False):
        super().__init__()
        # self.performer = performer
        # if performer:
        #     self.model = PerformerEncDec(num_tokens=vocab_size, max_seq_len=max(output_length, input_length), depth=n_layers, heads=n_heads, dim=d_model, ff_mult=d_filter//d_model, dim_head=d_model//n_heads)
        # else:
        #     self.model = Transformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_filter=d_filter)
        self.model = RewardTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_filter=d_filter)
        # Summarization loss
        criterion = nn.CrossEntropyLoss(reduce='none')
        self.loss_fn = lambda pred,target,mask: (criterion(pred.permute(0,2,1),target)*mask).sum()/mask.sum()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def forward(self,batch,optimize=True):
        target,mask = batch['target_sequence'],batch['decoder_mask']
        # if self.performer:
        #     _, pred_logits = self.model(**batch)
        #     target = target[:,1:]
        #     mask = mask[:, 1:]
        # else:
        #     pred_logits = self.model(**batch)
        pred_logits = self.model(**batch)
        loss = self.loss_fn(pred_logits,target,mask)
        accuracy = (th.eq(pred_logits.argmax(dim=2,keepdim=False),target).float()*mask).sum()/mask.sum()
        
        if optimize:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
        return loss, accuracy