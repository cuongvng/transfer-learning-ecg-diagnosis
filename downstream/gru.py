import torch
import torch.nn as nn

class GRU_Classifier(nn.Module):
	def __init__(
			self, 
			num_classes, 
			device, 
			n_features=12, 
			seq_len=1000, 
			hidden_size=100, 
			dropout=0.3
):
		super(GRU_Classifier, self).__init__()
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.device = device

		self.gru = nn.GRU(input_size=n_features, hidden_size=self.hidden_size,
						  batch_first=True, dropout=dropout)

		self.fc = nn.Linear(in_features=seq_len*self.hidden_size, out_features=self.num_classes)

	def forward(self, X):
		h0 = self.init_hidden(X)
		X, hn = self.gru(X, h0)
		X = torch.flatten(X, start_dim=1)

		# Free up GPU memory
		del h0, hn
		torch.cuda.empty_cache()

		return self.fc(X)

	def init_hidden(self, X):
		h0 = torch.zeros((1, X.shape[0], self.hidden_size)).to(self.device, dtype=torch.double)
		return h0
