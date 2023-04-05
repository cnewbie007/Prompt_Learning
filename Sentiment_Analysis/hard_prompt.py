import os
import copy
import torch
import argparse
from utils import *
warnings.filterwarnings("ignore")
from datetime import datetime
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, RobertaForMaskedLM


class HardPrompt:
    def __init__(self, prompt):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
        self.prompt = prompt
        self.pos_prefix = self.prompt + args().pos_token + '. '
        self.neg_prefix = self.prompt + args().neg_token + '. '
        self.pos_token_id = self.tokenizer(self.pos_prefix, return_tensors='pt')['input_ids'][0, 5]
        self.neg_token_id = self.tokenizer(self.neg_prefix, return_tensors='pt')['input_ids'][0, 5]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args().learning_rate)
        self.loaders = amz_review_ds(
            num_train=args().num_train,
            num_val=args().num_val,
            num_test=args().num_test,
            train_batchsz=args().train_batchsz,
            test_batchsz=args().test_batchsz,
            case_name='hard_prompt',
            prompt=prompt,
            pos_token=args().pos_token,
            neg_token=args().neg_token
        )
        print(
            ' #########################################', '\n',
            '######## Data Initialization Done #######'.format(device), '\n',
            '#########################################',
        )

    def pred_decode(self, logits, mask_token_index):
        # Get the predicted label by comparing the probability of positive tokens and negative tokens
        batch_pred = []
        output_size = len(logits)
        for i in range(output_size):
            pos_prob = logits[i, mask_token_index][0, self.pos_token_id]
            neg_prob = logits[i, mask_token_index][0, self.neg_token_id]
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

    def evaluation(self):
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.loaders['test']):
                review, label, target = batch['review'], batch['label'], batch['target']
                # encode the evaluation input
                encoded_inputs = self.tokenizer(
                    review,
                    truncation=True,
                    padding=True,
                    max_length=150,
                    return_tensors='pt',
                    add_special_tokens=True
                ).to(device)
                # get the output
                outputs = self.model(**encoded_inputs).logits
                mask_token_id = (encoded_inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

                # append the predictions and the target labels
                pred_tgt = self.pred_decode(outputs, mask_token_id)
                y_pred += pred_tgt
                y_true += target
                # print(pred_word)
                # print(pred_tgt)
                # print(target)
                # print()
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy

    # def validation(self):

    def encode_labels(self, input_encoding):
        batch_max_length = len(input_encoding[0])
        batch_num_sample = len(input_encoding)
        for i in range(batch_num_sample):
            temp = torch.full((batch_max_length,), -100, dtype=torch.long)
            temp[5] = input_encoding[i][5]
            temp = temp.to(device)
            input_encoding[i] = temp
        return input_encoding

    def train(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.loaders['train']):
            review, label, target = batch['review'], batch['label'], batch['target']

            # encode review and labels
            encoded_input = self.encode_inputs(review)

            encoded_label = self.encode_inputs(label)
            encoded_label = self.encode_labels(encoded_label)
            # predict and compute loss
            output = self.model(encoded_input, labels=encoded_label)
            loss = output.loss
            epoch_loss += loss.item()
            # print('Loss: {:.5f}'.format(loss.item()))

            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss / len(self.loaders['train'])

    def workflow(self):
        # best_loss = float('inf')
        # best_accuracy = 0
        for epoch in range(args().num_epochs):
            # train one epoch
            train_loss = self.train()

            accuracy = self.evaluation()

            print()
            print(
                '*Epoch: {:02d}/{:02d}'.format(epoch + 1, args().num_epochs), '\n',
                'Train Loss: {:.5f}'.format(train_loss), '\n',
                'Accuracy: {}'.format(accuracy)
            )

            with open('./logs/HardPrompt_RoBERTa_AMZ_review.txt', 'a') as f:
                f.write('Epoch: {:02d}/{:02d} | '.format(epoch + 1, args().num_epochs))
                f.write('Train Loss: {:.5f} | '.format(train_loss))
                f.write('Accuracy: {:.5f}'.format(accuracy))
                f.write('\n')
            # if val_loss < best_loss:
            #     accuracy = self.evaluation()
            #     best_accuracy = max(accuracy, best_accuracy)
            #     print(
            #         '*Current Accuracy: {:.5f}'.format(accuracy), '\n',
            #         'Best Accuracy: {:.5f}'.format(best_accuracy)
            #     )


# Arguments
def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--num_train", type=int, default=5000,
                                  help="batch size")
    train_arg_parser.add_argument("--num_val", type=int, default=1000,
                                  help="batch size")
    train_arg_parser.add_argument("--num_test", type=int, default=1000,
                                  help="batch size")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=8,
                                  help="batch size")
    train_arg_parser.add_argument("--test_batchsz", type=int, default=4,
                                  help="batch size")
    train_arg_parser.add_argument("--num_epochs", type=int, default=3,
                                  help="batch size")
    train_arg_parser.add_argument("--learning_rate", type=float, default=2e-5,
                                  help=" Total Epochs ")
    train_arg_parser.add_argument("--pos_token", type=str, default='good',
                                  help=" Total Epochs ")
    train_arg_parser.add_argument("--neg_token", type=str, default='bad',
                                  help=" Total Epochs ")
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

    template = 'Overall, it was '

    with open('./logs/HardPrompt_RoBERTa_AMZ_review.txt', 'a') as file:
        file.write('##############################################')
        file.write('\n')
        file.write('Prompt: {} <mask>'.format(template))
        file.write('\n')
        for key in args().__dict__.keys():
            file.write('{}: {}'.format(key, args().__dict__[key]))
            file.write('\n')

    HardPrompt(template).workflow()
