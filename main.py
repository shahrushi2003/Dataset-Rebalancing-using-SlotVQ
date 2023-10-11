from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import wandb
import torch
from tqdm.auto import tqdm
import numpy as np
from random import choices
from collections import Counter
import random
import pandas as pd
from data_prep import Rebalanced_Dataset
from torchvision import transforms
import copy
import argparse
import os
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from data_prep import MnistTrain
from model import Lightning_AE
from slot_collector import Slot_Collector, apply_temp
from classifier import Lightning_Classifier
from configs import model_args, data_args, classifier_args

def get_data(data_args):
	seed_everything(seed=data_args["seed"], workers=True)
	data_prepper = MnistTrain(args = data_args)
	data_prepper.prepare_data_loaders()
	return data_prepper

def train_slot_model(model_args, data_prepper, run_name):
	seed_everything(seed=model_args["seed"], workers=True)
	model = Lightning_AE(model_args, data_prepper.train_loader).to(device)
	if model_args["use_kmeans"]:
		model.model.warmup_quantizers()
	summary = ModelSummary(model, max_depth=-1)
	print("Model Summary:")
	print(summary)
	print("Configs")
	print(model_args)

	# model.model.eval()
	# coll = Slot_Collector(model_args["num_slots"], model_args["codebook_size"])
	# coll.get_slots(data_prepper.train_loader, model)
	# # print("\nSlots Distribution for Epoch:", self.current_epoch)
	# print([len(coll.slot_collector[i]) for i in coll.slot_collector])

	checkpoint_callback = ModelCheckpoint(monitor='total_val_loss', mode='min')

	wandb_logger = WandbLogger(project='Slot_Rebalancing_Experiments',
							name=run_name,
							log_model='all')
	# log gradients, parameter histogram and model topology
	wandb_logger.watch(model, log="all")

	trainer = Trainer(accelerator="gpu",
					logger=wandb_logger,
					max_epochs=model_args["num_epochs"],
					callbacks=[checkpoint_callback],
					)

	trainer.fit(model, train_dataloaders=data_prepper.train_loader, val_dataloaders=data_prepper.val_loader)
	# wandb.finish()
	return model

def plot_images(slots):
	output_folder = "./"
	num_images_to_show = 20
	slot_dict = {}

	for i in slots:
		if len(slots[i]) != 0:
			slot_dict[i] = slots[i]

	fig, axs = plt.subplots(len(slot_dict), num_images_to_show+1, figsize=(num_images_to_show, len(slot_dict)))

	# Set the aspect ratio to be equal for correct image display
	for i in range(len(slot_dict)):
		for j in range(num_images_to_show+1):
			axs[i, j].axis('off')
	c = 0

	# Loop through each slot and its images
	for slot, images in slot_dict.items():
		# Print the slot number on the left side of the row
		axs[c, 0].text(0, 0.5, f"Bin {slot}", fontsize=12, va='center', ha='center')
		axs[c, 0].axis('off')

		# Take up to num_images_to_show images for the slot
		if len(images) > num_images_to_show:
			images = random.sample(images, num_images_to_show)

		for i, image_info in enumerate(images):
			if image_info is not None:
				image_path, target = image_info
				img = cv2.imread(image_path)  # Read the image using cv2
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
				axs[c, i + 1].imshow(img)

		c += 1

	# Save the plot as an image in the output folder
	output_file_path = os.path.join(output_folder, "slot_visualization.png")
	# plt.axis('off')
	plt.savefig(output_file_path)

	# Show the plot (optional)
	plt.show()

	# Close the plot to release resources (optional)
	plt.close()


def rebalance(model, data_prepper, model_args):
	# disable randomness, dropout, etc...
	seed_everything(seed=model_args["seed"], workers=True)
	model.eval()

	coll = Slot_Collector(model_args["num_slots"], model_args["codebook_size"])
	coll.get_slots(data_prepper.train_loader, model)

	print("Slots Successfully Collected!")

	slots = coll.slot_collector

	plot_images(slots)

	for i in coll.slot_collector:
		print(len(coll.slot_collector[i]), end=" ")
	print()
	print("The total number of slots is", len(slots))

	probs = np.array([len(slots[i]) for i in slots])

	# Without temperature
	new_probs = apply_temp(model_args["temperature"], probs)

	# Sanity Check 1
	print("Sanity Check 1", np.sum(apply_temp(1, probs)) == 1)

	population = list(range(len(probs)))
	weights = np.array(new_probs)

	samples = choices(population, weights, k=len(data_prepper.train_dataset))

	for i in dict(Counter(samples)):
		print(dict(Counter(samples))[i], new_probs[i]*len(data_prepper.train_dataset), probs[i])

	new_imgs = []
	new_labels = []
	counts = dict(Counter(samples))
	for i in counts:
		num_samples = counts[i]
		curr_samples = random.choices(slots[i], k=num_samples)
		new_imgs += [path for path, label in curr_samples]
		new_labels += [label.item() for path, label in curr_samples]



	df = pd.DataFrame(list(zip(new_imgs, new_labels)),
					columns =['Path', 'Label'])
	df.to_csv(f'balanced_data_beta{model_args["beta"]}.csv')
	print(df.head(100))

	# Sanity Check 2
	print("Sanity Check 2", len(df)==len(data_prepper.train_dataset))

	rebalanced_data = Rebalanced_Dataset(dataframe=df, transform = transforms.ToTensor())
	return rebalanced_data



def eval_using_classifier(data_prepper, rebalanced_data, classifier_args, run_name):
	# Initialising models
	seed_everything(seed=classifier_args["seed"], workers=True)
	model_1 = Lightning_Classifier(classifier_args, data_type="Original").to(device)
	model_2 = Lightning_Classifier(classifier_args, data_type="Rebalanced").to(device)
	model_2.model = copy.deepcopy(model_1.model).to(device)


	wandb_logger = WandbLogger(project='Slot_Rebalancing_Experiments',
				   name=run_name,
				   log_model='all')

	trainer = Trainer(profiler="simple",
					deterministic=True,
					accelerator="gpu",
					logger=wandb_logger,
					max_epochs=classifier_args["num_epochs"],
					# callbacks=[checkpoint_callback],
					)


	trainer.fit(model_1, train_dataloaders=data_prepper.train_loader, val_dataloaders=[data_prepper.val_loader, data_prepper.test_loader])

	trainer = Trainer(profiler="simple",
					deterministic=True,
					accelerator="gpu",
					logger=wandb_logger,
					max_epochs=classifier_args["num_epochs"],
					# callbacks=[checkpoint_callback],
					)
	balanced_data_loader = DataLoader(rebalanced_data, batch_size=classifier_args["batch_size"], shuffle=True, num_workers=classifier_args["num_workers"])
	trainer.fit(model_2, train_dataloaders=balanced_data_loader, val_dataloaders=[data_prepper.val_loader, data_prepper.test_loader])
	wandb.finish()

# driver code
if __name__ == "__main__":

	# set up args for easy tinkering
	def get_args():
		parser = argparse.ArgumentParser()
		parser.add_argument("--slot-run-name", default="")
		parser.add_argument("--orig_classifier-run-name", default="")
		parser.add_argument("--rebal_classifier-run-name", default="")
		parser.add_argument("--dataset-name", default="mnist", choices=["mnist"])
		parser.add_argument("--logging", default="off")
		args = parser.parse_args()
		return args

	# args = get_args()
	run_name = f'seed{model_args["seed"]}_alpha{model_args["alpha"]}_beta{model_args["beta"]}_with_kmeans'
	data_prepper = get_data(data_args)
	model = train_slot_model(model_args, data_prepper, run_name)
	rebalanced_data = rebalance(model, data_prepper, model_args)
	eval_using_classifier(data_prepper, rebalanced_data, classifier_args, run_name)

