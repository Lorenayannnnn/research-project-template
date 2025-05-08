from dataclasses import dataclass

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler


@dataclass
class CustomTrainerConfig():
    model: None
    lr: None
    train_dataloader: None
    eval_dataloader: None
    test_dataloader: None
    num_epochs: None
    batch_size: None
    micro_batch_size: None


class CustomTrainer():
    def __init__(self, config: CustomTrainerConfig):
        super(CustomTrainer, self).__init__()
        self.model = config.model
        self.lr = config.lr
        self.train_dataloader = config.train_dataloader
        self.test_dataloader = config.test_dataloader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.micro_batch_size = config.micro_batch_size

        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        self.accelerator, self.train_dataloader, self.model, self.optimizer, self.scheduler = self.setup_accelerator()

        self.test_dataloader = config.test_dataloader

    def setup_optimizer(self):
        """
        return:
            - optimizer: torch.optim
        """
        # ===================================================== #
        # Task: Initialize the loss function for action predictions
        # and target predictions. Also initialize your optimizer.
        # ===================================================== #
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        return optimizer

    def setup_scheduler(self):
        num_training_steps = self.num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        return lr_scheduler

    def setup_accelerator(self):
        accelerator = Accelerator(gradient_accumulation_steps=self.batch_size // self.micro_batch_size)
        train_dataloader, model, optimizer, scheduler = accelerator.prepare(self.train_dataloader, self.model,
                                                                            self.optimizer,
                                                                            self.scheduler)
        return accelerator, train_dataloader, model, optimizer, scheduler

    def train(self):
        num_training_steps = len(self.train_dataloader) * self.num_epochs
        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if step % (self.batch_size // self.micro_batch_size) == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                progress_bar.update(1)

    def eval(self):
        progress_bar = tqdm(range(len(self.eval_dataloader)))
        self.model.eval()
        for batch in self.eval_dataloader:
            batch = {k: v.to() for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            progress_bar.update(1)

    def predict(self):
        progress_bar = tqdm(range(len(self.test_dataloader)))
        for batch in self.test_dataloader:
            batch = {k: v.to() for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            self.optimizer.zero_grad()
            progress_bar.update(1)
