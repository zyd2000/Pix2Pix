from torch import optim
from torch.utils.data import DataLoader
import config
from dataset import Dataset
from utils import save_some_examples, load_checkpoint
from generator_model import Generator


def main():
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    test_dataset = Dataset(root_dir="data/" + config.train_type + "/test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    save_some_examples(gen, test_loader, 1, folder="test")


if __name__ == "__main__":
    main()
