import os
import torch
import argparse
import torch.nn as nn
from utils import *
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, RobertaForMaskedLM
warnings.filterwarnings("ignore")


class SoftPrompt:
    def __init__(self, prompt):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
        self.num_soft_tokens = args().num_soft_tokens
        self.prompt_emb = torch.randn((self.num_soft_tokens, 768), requires_grad=True)
        self.pos_token_id = self.tokenizer(' {}'.format(args().pos_token), return_tensors='pt')['input_ids'][0, 1]
        self.neg_token_id = self.tokenizer(' {}'.format(args().neg_token), return_tensors='pt')['input_ids'][0, 1]
        self.init_soft_embedding(prompt)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args().learning_rate)
        self.pmt_opti = torch.optim.Adam([self.prompt_emb], lr=args().pmt_learning_rate)
        self.loaders = amz_review_ds(
            num_train=args().num_train,
            num_val=args().num_val,
            num_test=args().num_test,
            train_batchsz=args().train_batchsz,
            test_batchsz=args().test_batchsz,
            case_name='soft_tune',
            prompt=prompt,
            pos_token=args().pos_token,
            neg_token=args().neg_token
        )
        print(
            ' #########################################', '\n',
            '######## Data Initialization Done #######'.format(device), '\n',
            '#########################################',
        )

    def init_soft_embedding(self, prompt):
        # if we want to set a hard prompt as our initial soft prompt embedding, extract the embedding
        if args().hard_initialization == 'yes':
            prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
            self.num_soft_tokens = prompt_tokens.shape[1] - 2
            prompt_embedding = self.model.get_input_embeddings()(prompt_tokens[:, 1:-1])
            self.prompt_emb = nn.Parameter(prompt_embedding, requires_grad=True)

    def pred_decode(self, logits, mask_token_index):
        """
        :param logits:
            The prediction score of the language model head (lm_head) with dimension: batchsz * num_tokens * |V|,
            we want to find the probability of positive and negative tokens from it.
        :param mask_token_index:
            The index of the <mask> token.
            Since we manipulate the embedding, the index is not related to the position in text input.
        :return:
            The corresponding predictions of the given batch.
            0 -> negative, 1 -> positive
        """
        batch_pred = []
        output_size = len(logits)
        for i in range(output_size):
            pos_prob = logits[i, mask_token_index][self.pos_token_id]
            neg_prob = logits[i, mask_token_index][self.neg_token_id]
            if pos_prob > neg_prob:
                batch_pred.append(1)
            else:
                batch_pred.append(0)

        return batch_pred

    def encode_inputs(self, input_data):
        temp = self.tokenizer.batch_encode_plus(
            input_data,
            truncation=True,
            padding=True,
            max_length=150,
            return_tensors='pt',
            add_special_tokens=True)
        temp = temp['input_ids'].to(device)
        return temp

    def concatenate_embedding(self, temp):
        """
        :param temp:
            The batch embedding before adding the soft prompt embedding.
        :return:
            The batch embedding after adding the soft prompt embedding.
        """
        # Initialize the output embedding tensor, should add the length of the soft prompt
        output = torch.empty(temp.shape[0], temp.shape[1] + self.num_soft_tokens, 768)
        size = len(temp)
        for i in range(size):
            start_token = temp[i:i + 1, 0:1]    # The start token embedding
            prompt_emb = self.prompt_emb.to(device)    # The prompt embedding
            input_emb = temp[i:i + 1, 1:]   # the <mask>. and the original input embedding
            if args().hard_initialization == 'yes':
                output[i] = torch.cat([start_token, prompt_emb, input_emb], dim=1)[0]
            else:
                output[i] = torch.cat([start_token, prompt_emb.unsqueeze(0), input_emb], dim=1)[0]
        output.requires_grad_(True)
        return output.to(device)

    def evaluation(self):
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.loaders['test']):
                review, label, target = batch['review'], batch['label'], batch['target']
                # encode the evaluation input
                encoded_input = self.tokenizer(
                    review,
                    truncation=True,
                    padding=True,
                    max_length=150,
                    return_tensors='pt',
                    add_special_tokens=True
                ).to(device)

                input_emb = self.model.get_input_embeddings()(encoded_input['input_ids'])
                overall_emb = self.concatenate_embedding(input_emb)

                # get the output
                outputs = self.model(inputs_embeds=overall_emb).logits
                mask_token_id = self.num_soft_tokens + 1

                # append the predictions and the target labels
                pred_tgt = self.pred_decode(outputs, mask_token_id)
                y_pred += pred_tgt
                y_true += target

        accuracy = accuracy_score(y_true, y_pred)

        return accuracy

    # def validation(self):

    def encode_labels(self, input_encoding):
        output = torch.zeros(
            (input_encoding.shape[0], input_encoding.shape[1] + self.num_soft_tokens),
            dtype=torch.long
        )
        batch_max_length = len(input_encoding[0]) + self.num_soft_tokens
        batch_num_sample = len(input_encoding)
        for i in range(batch_num_sample):
            temp = torch.full((batch_max_length,), -100, dtype=torch.long)
            # define the label at the position of <mask> after concatenate the soft prompt
            temp[self.num_soft_tokens + 1] = input_encoding[i][1]
            output[i] = temp
        return output.to(device)

    def train(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.loaders['train']):
            review, label, target = batch['review'], batch['label'], batch['target']

            # encode review and labels
            encoded_input = self.encode_inputs(review)
            input_emb = self.model.get_input_embeddings()(encoded_input)
            overall_emb = self.concatenate_embedding(input_emb)
            encoded_label = self.encode_inputs(label)
            encoded_label = self.encode_labels(encoded_label)

            # predict and compute loss
            output = self.model(inputs_embeds=overall_emb, labels=encoded_label)
            loss = output.loss
            epoch_loss += loss.item()

            # optimize the model and the soft embedding
            if args().model_tuning == 'yes':
                self.optimizer.zero_grad()
            if args().soft_tuning == 'yes':
                self.pmt_opti.zero_grad()

            loss.backward()

            if args().model_tuning == 'yes':
                self.optimizer.step()
            if args().soft_tuning == 'yes':
                self.pmt_opti.step()

        return epoch_loss / len(self.loaders['train'])

    def workflow(self):

        for epoch in range(args().num_epochs):
            # train one epoch
            train_loss = self.train()

            accuracy = self.evaluation()

            print()
            print(
                '*Epoch: {:02d}/{:02d}'.format(epoch + 1, args().num_epochs), '\n',
                'Train Loss: {:.5f}'.format(train_loss), '\n',
                'Accuracy: {}'.format(accuracy)
                # 'Validation Loss: {:.5f}'.format(val_loss)
            )

            with open('./logs/Soft_Prompt_RoBERTa_AMZ_review.txt', 'a') as f:
                f.write('Epoch: {:02d}/{:02d} | '.format(epoch + 1, args().num_epochs))
                f.write('Train Loss: {:.5f} | '.format(train_loss))
                f.write('Accuracy: {:.5f}'.format(accuracy))
                f.write('\n')


# Arguments
def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--num_train", type=int, default=100,
                                  help=" number training samples")
    train_arg_parser.add_argument("--num_val", type=int, default=20,
                                  help=" number of validation samples")
    train_arg_parser.add_argument("--num_test", type=int, default=2000,
                                  help=" number of test samples")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=32,
                                  help=" training batch size ")
    train_arg_parser.add_argument("--test_batchsz", type=int, default=32,
                                  help=" testing batch size ")
    train_arg_parser.add_argument("--num_epochs", type=int, default=100,
                                  help=" training epochs ")
    train_arg_parser.add_argument("--num_soft_tokens", type=int, default=10,
                                  help=" number of soft prompt embeddings ")
    train_arg_parser.add_argument("--learning_rate", type=float, default=1e-5,
                                  help=" pre-trained model learning rate ")
    train_arg_parser.add_argument("--pmt_learning_rate", type=float, default=1e-4,
                                  help=" soft prompt embedding learning rate ")
    train_arg_parser.add_argument("--pos_token", type=str, default='great',
                                  help=" the positive label token ")
    train_arg_parser.add_argument("--neg_token", type=str, default='terrible',
                                  help=" the negative label token ")
    train_arg_parser.add_argument("--hard_initialization", type=str, default='no',
                                  help=" take a hard template as the initial soft embedding ")
    train_arg_parser.add_argument("--model_tuning", type=str, default='no',
                                  help=" tune the model parameters or not ")
    train_arg_parser.add_argument("--soft_tuning", type=str, default='yes',
                                  help=" tune the soft prompt embedding or not (should be yes) ")
    return train_arg_parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(args().gpu) if torch.cuda.is_available() else 'cpu')

    print(
        ' #########################################', '\n',
        '############ HyperParameters: ###########'.format(device), '\n',
        '#########################################',
    )

    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    for key in args().__dict__.keys():
        print(key + ':', args().__dict__[key])
    print()

    error = (args().model_tuning == 'yes') + (args().soft_tuning == 'yes')
    assert error > 0, "You have to update at least one set of the parameters! Model or Soft Prompt Embedding."

    template = 'Overall, it was'

    with open('./logs/Soft_Prompt_RoBERTa_AMZ_review.txt', 'a') as file:
        file.write('##############################################')
        file.write('\n')
        file.write('Prompt: {} <mask>'.format(template))
        file.write('\n')
        for key in args().__dict__.keys():
            file.write('{}: {}'.format(key, args().__dict__[key]))
            file.write('\n')

    SoftPrompt(template).workflow()
