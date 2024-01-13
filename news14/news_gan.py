import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from sklearn import metrics
import sys

class_list = ['财经', '科技', '时政', '房产', '社会', '游戏', '家居', '时尚', '股票', '彩票', '娱乐', '教育', '星座', '体育']
MAX_VOCAB_SIZE = 10000
num_class = len(class_list)
UNK, PAD = '<UNK>', '<PAD>'
torch.cuda.set_device(0)

if not os.path.exists('./train_process.txt'):
    with open('./train.txt', 'r', encoding='UTF-8') as f:
        with open('./train_process.txt', 'w', encoding='UTF-8') as f2:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                f2.write(content + '\t' + str(class_list.index(label)) + '\n')

if not os.path.exists('./dev_process.txt'):
    with open('./dev.txt', 'r', encoding='UTF-8') as f:
        with open('./dev_process.txt', 'w', encoding='UTF-8') as f2:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                f2.write(content + '\t' + str(class_list.index(label)) + '\n')

class Config(object):
    def __init__(self, dataset, embedding):
        self.model_name = 'news_GAN'
        self.class_list = class_list
        self.vocab_path = os.path.join(dataset, './vocab.pkl')
        self.train_path = os.path.join(dataset, './train_process.txt')
        self.dev_path = os.path.join(dataset, './dev_process.txt')
        self.test_path = os.path.join(dataset, './test.txt')
        self.save_path = os.path.join(dataset, './', self.model_name + '.ckpt')
        self.log_path = os.path.join(dataset, 'log', self.model_name)
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(dataset, 'data', embedding))["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 10
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_vocab, 300, padding_idx=num_vocab - 1)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(9600, num_class)

    def forward(self, x):
        if(isinstance(x, tuple)):
            x = self.embedding(x[0])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 32 * 300)  # Assuming input noise size is 100

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 300)
        return x

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def evaluate(config, discriminator, generator, data_iter, test=False):
    discriminator.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = discriminator(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def train(config, discriminator, generator, train_iter, dev_iter):
    discriminator.train()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = discriminator(trains)
            discriminator.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            random_noise = torch.randn(128, 100).to(device)  # Batch size of 128, noise size of 100
            # random_noise = torch.randint(0, 100, (128,)).to(device)
            fake_data = generator(random_noise)

            fake_labels = torch.randint(0, num_class, (128,)).to(device)  # Convert fake_labels to Long data type

            fake_outputs = discriminator(fake_data.detach())
            fake_loss = F.cross_entropy(fake_outputs, fake_labels)
            fake_loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, discriminator, generator, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(discriminator.state_dict(), "news_GAN.ckpt")

                msg = '\rIter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc), end='')
                discriminator.train()
            total_batch += 1
        print()

def model_test(config, discriminator, generator, test_iter):
    discriminator.load_state_dict(torch.load(config.save_path))
    discriminator.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, discriminator, generator, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))

def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')
    else:
        tokenizer = lambda x: [y for y in x]
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    # test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

if __name__ == "__main__":
    test_flag = False
    if len(sys.argv) == 2:
        if sys.argv[1] == '--test':
            test_flag = True
            print("Test mode")
    dataset = ''
    model_name = 'GAN'
    embedding = 'random'
    config = Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    print("Loading data...")
    vocab, train_data, dev_data = build_dataset(config, False)
    num_vocab = len(vocab)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    time_dif = get_time_dif(start_time)

    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    discriminator = Discriminator().to(config.device)
    generator = Generator().to(config.device)
    init_network(discriminator)
    # print(discriminator.parameters)

    if not test_flag:
        train(config, discriminator, generator, train_iter, dev_iter)
    model_test(config, discriminator, generator, dev_iter)