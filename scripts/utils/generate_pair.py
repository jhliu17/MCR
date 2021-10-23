from matchzoo.dataloader import InstanceDataset
from matchzoo import DataPack


save_dir = "./dataset/amazon/clothing/"
dup = 1
neg = 14
interval = 1
max_pos_samples = None

train_pack_processed = DataPack.load(save_dir, 'train')
InstanceDataset.generate_dataset(
    train_pack_processed,
    mode='pair',
    num_dup=dup,
    num_neg=neg,
    max_pos_samples=max_pos_samples,
    save_dir=save_dir,
    building_interval=interval,
    name='train'
)
