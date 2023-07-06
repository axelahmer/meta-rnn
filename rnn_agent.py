import torch
import torch.nn as nn

class PolicyRNNCell(nn.Module):
    def __init__(self, size, hidden_size):
        super(PolicyRNNCell, self).__init__()

        self.i2h = nn.Linear(size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2i = nn.Linear(hidden_size, size)
        self.h2c = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        hidden_prime = self.tanh(self.i2h(input) + self.h2h(hidden))
        next_input = self.h2i(hidden_prime)
        confidence = self.h2c(hidden_prime)
        return next_input, confidence, hidden_prime

class RNNAgent(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_size, iters=3):
        super().__init__()

        self.iters = iters

        self.obs2i = nn.Linear(observation_size, num_actions)
        self.obs2h = nn.Linear(observation_size, hidden_size)
        self.rnn = PolicyRNNCell(num_actions, hidden_size)

        # self.embedding = nn.Embedding(iters, hidden_size)

    def forward(self, observation):
        input = self.obs2i(observation)
        hidden = self.obs2h(observation)

        logits = [input]
        confidences = []
        for _ in range(self.iters):
            output, confidence, hidden = self.rnn(input, hidden)
            input = input + output
            logits.append(input)
            confidences.append(confidence)

        logits = torch.stack(logits)[:-1]
        confidences = torch.stack(confidences)

        soft_confs = torch.nn.functional.softmax(confidences, dim=0)
        logits = torch.sum(soft_confs * logits, dim=0)

        # create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)

        return dist, soft_confs.flatten()