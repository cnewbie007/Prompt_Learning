import random
import datasets
import warnings
import logging
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)


def ft_transfer(temp_data, temp_list):
    temp_data = temp_data[temp_list]
    temp_dataset = datasets.Dataset.from_dict(
        {
            'review': temp_data['review'],
            'label': temp_data['label'],
        }
    )
    return temp_dataset


def hard_transfer(temp_data, temp_list, prompt, pos_token, neg_token):
    positive = prompt + pos_token + '. '
    negative = prompt + neg_token + '. '
    temp_data = temp_data[temp_list]
    size = len(temp_data['label'])
    target = []

    for i in range(size):
        review = temp_data['review'][i]
        label = temp_data['label'][i]
        if label == 1:
            target.append(1)
            temp_data['label'][i] = positive + review
        else:
            target.append(0)
            temp_data['label'][i] = negative + review
        temp_data['review'][i] = prompt + '<mask>. ' + temp_data['review'][i]
    temp_dataset = datasets.Dataset.from_dict(
        {
            'review': temp_data['review'],
            'label': temp_data['label'],
            'target': target
        }
    )
    return temp_dataset


def soft_transfer(temp_data, temp_list, pos_token, neg_token):
    positive = ' ' + pos_token + '. '
    negative = ' ' + neg_token + '. '
    temp_data = temp_data[temp_list]
    size = len(temp_data['label'])
    target = []

    for i in range(size):
        review = temp_data['review'][i]
        label = temp_data['label'][i]
        if label == 1:
            target.append(1)
            temp_data['label'][i] = positive + review
        else:
            target.append(0)
            temp_data['label'][i] = negative + review
        temp_data['review'][i] = ' <mask>. ' + temp_data['review'][i]
    temp_dataset = datasets.Dataset.from_dict(
        {
            'review': temp_data['review'],
            'label': temp_data['label'],
            'target': target
        }
    )
    return temp_dataset


def amz_review_ds(
        num_train, num_val, num_test, train_batchsz, test_batchsz, case_name, prompt, pos_token, neg_token, seed=7
):

    # edit the target label
    dataset = datasets.load_dataset('amazon_reviews_multi', 'en', split='train')
    stars = dataset['stars']
    category = dataset['product_category']

    size = len(stars)
    labels = []

    data_index = []
    # preprocess the prompt label
    for i in range(size):
        domain = category[i]
        if domain == 'home':
            data_index.append(i)
        curr_star = stars[i]
        if curr_star > 3:
            labels.append(1)
        else:
            labels.append(0)

    # get the data index for different sets
    splits = {}
    random.seed(seed)
    random.shuffle(data_index)
    splits['train'] = data_index[:num_train]
    splits['val'] = data_index[-(num_val + num_test):-num_test]
    splits['test'] = data_index[-num_test:]

    ds = datasets.Dataset.from_dict(
        {
            'review': dataset['review_body'],
            'label': labels,
        }
    )

    # build the data and the dataloader
    modes = ['train', 'val', 'test']
    data = {}
    loaders = {}
    for mode in modes:
        # transfer the selected data to dataset
        if case_name == 'hard_prompt':
            data[mode] = hard_transfer(ds, splits[mode], prompt, pos_token, neg_token)
        if case_name == 'fine_tune':
            data[mode] = ft_transfer(ds, splits[mode])
        if case_name == 'soft_tune':
            data[mode] = soft_transfer(ds, splits[mode], pos_token, neg_token)
        if mode != 'test':
            loaders[mode] = DataLoader(data[mode], batch_size=train_batchsz, shuffle=True)
        else:
            loaders[mode] = DataLoader(data[mode], batch_size=test_batchsz, shuffle=False)

    return loaders
