from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

class DatasetDataModule(LightningDataModule):
    def __init__(self, dataset, tokenizer, out_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.out_dim = out_dim
        

    def setup(self, stage=None):
        dataset_class = self.dataset['dataset_class']
        self.train_set = dataset_class(self.dataset['train_file'], self.tokenizer, self.out_dim)
        self.validate_set = (
            dataset_class(self.dataset['validate_file'], self.tokenizer, self.out_dim)
            if self.dataset['validate_file']
            else None
        )
        self.test_set = (
            dataset_class(self.dataset['test_file'], self.tokenizer, self.out_dim)
            if self.dataset['test_file']
            else None
        )

    # TODO: set optimal num_workers for local and cloud training
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate_set, batch_size=self.batch_size) \
            if self.validate_set else None

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size) \
            if self.test_set else None