import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import diags, csr_matrix
from tqdm import tqdm


def Word2Vec(sentences, vector_size=100, window=5, epochs=5, sg=1, hs=1, min_count=0, workers=4, device='cuda'):
    """
    A PyTorch implementation of Word2Vec function, mimicking gensim's interface.
    Args:
        sentences: List of tokenized sentences (e.g., walks).
        vector_size: Dimensionality of the embedding vectors.
        window: Context window size.
        epochs: Number of training iterations.
        sg: Skip-Gram (1) or CBOW (0). Only Skip-Gram is implemented.
        hs: Hierarchical softmax. Ignored in this implementation.
        min_count: Minimum frequency count for words to be included in the vocabulary.
        workers: Number of workers (used for DataLoader, not parallel training).
        device: Device for training ('cuda' or 'cpu').

    Returns:
        A mock gensim Word2Vec object with a similar interface.
    """
    class Word2VecMock:
        def __init__(self, idx2word, embeddings):
            self.wv = self.WordVectors(idx2word, embeddings)

        class WordVectors:
            def __init__(self, idx2word, embeddings):
                self.index_to_key = idx2word
                self.vectors = embeddings
                self.key_to_index = {word: idx for idx, word in enumerate(idx2word)}

            def __getitem__(self, key):
                return self.vectors[self.key_to_index[key]]

    # Build vocabulary
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            vocab[word] = vocab.get(word, 0) + 1
    vocab = {word: count for word, count in vocab.items() if count >= min_count}
    idx2word = list(vocab.keys())
    word2idx = {word: idx for idx, word in enumerate(idx2word)}

    vocab_size = len(idx2word)
    
    # Unigram distribution for negative sampling
    word_counts = torch.tensor([vocab[word] for word in idx2word], dtype=torch.float)
    unigram_distribution = word_counts / torch.sum(word_counts)
    unigram_distribution = unigram_distribution ** (3 / 4)
    unigram_distribution = unigram_distribution / torch.sum(unigram_distribution)

    # Create dataset and dataloader
    class Word2VecDataset(Dataset):
        def __init__(self, sentences, word2idx, window_size):
            self.pairs = []
            for sentence in sentences:
                indices = [word2idx[word] for word in sentence]
                for i, center in enumerate(indices):
                    # Define context window
                    window = indices[max(0, i - window_size):i] + indices[i+1:i+1+window_size]
                    for context in window:
                        self.pairs.append((center, context))

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            return self.pairs[idx]

    dataset = Word2VecDataset(sentences, word2idx, window_size=window)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=workers)

    # Define model
    class SkipGramNegSampling(nn.Module):
        def __init__(self, vocab_size, embedding_dim, num_negatives):
            super(SkipGramNegSampling, self).__init__()
            self.embedding_v = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_u = nn.Embedding(vocab_size, embedding_dim)
            self.num_negatives = num_negatives

        def forward(self, center_words, context_words, negative_words):
            center_embeds = self.embedding_v(center_words)
            context_embeds = self.embedding_u(context_words)
            negative_embeds = self.embedding_u(negative_words)

            positive_score = torch.sum(center_embeds * context_embeds, dim=1)
            positive_loss = torch.log(torch.sigmoid(positive_score))

            negative_score = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze(2)
            negative_loss = torch.log(torch.sigmoid(-negative_score)).sum(1)

            loss = - (positive_loss + negative_loss)
            return loss.mean()

    model = SkipGramNegSampling(vocab_size=vocab_size, embedding_dim=vector_size, num_negatives=5)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for center_words, context_words in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            center_words = torch.LongTensor(center_words).to(device)
            context_words = torch.LongTensor(context_words).to(device)
            negative_words = torch.multinomial(unigram_distribution, 
                                               num_samples=center_words.size(0)*5,
                                               replacement=True).view(center_words.size(0), 5).to(device)
            optimizer.zero_grad()
            loss = model(center_words, context_words, negative_words)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}")

    embeddings = model.embedding_v.weight.data.cpu().numpy()
    return Word2VecMock(idx2word, embeddings)
