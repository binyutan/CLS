from setting import model_map, dataset_map, parse_opts

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import nibabel as nib

from utils.file_process import load_lines, ensure_dir
from utils.metrics import seg_eval, compute_fnr_loss, apply_threshold


if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # Load model from checkpoint
    checkpoint = torch.load(sets.resume_path)
    net, _, _, _ = model_map(sets.model).generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # Create DataLoader
    testing_data = dataset_map(sets.dataset)(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=False)

    # Load file names
    img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    label_names = [info.split(" ")[1] for info in load_lines(sets.img_list)]

    # Generate probability maps using the model
    masks = model_map(sets.model).test(data_loader, net, img_names, sets)

    Nimg = len(img_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])

    # Ensure output directory exists
    save_dir = os.path.join("res/probability_matrix", sets.model, sets.dataset)
    ensure_dir(save_dir)

    for idx in range(Nimg):
        base_name = os.path.splitext(os.path.basename(img_names[idx]))[0]
        save_path = os.path.join(save_dir, base_name + ".npy")

        # Save probability map
        prob_map = masks[idx] if isinstance(masks[idx], np.ndarray) else masks[idx].cpu().numpy()
        np.save(save_path, prob_map)

        # Load ground truth label
        label = nib.load(os.path.join(sets.data_root, label_names[idx])).get_fdata()
        if label.ndim == 4 and label.shape[-1] == 1:
            label = np.squeeze(label, axis=-1)

        # Binarize prediction at heuristic threshold t=0.5 and evaluate Dice
        pred = apply_threshold(prob_map, t=0.5)
        dices[idx, :] = seg_eval(pred, label, range(sets.n_seg_classes))

    # Print mean Dice score for each class (excluding background)
    for idx in range(1, sets.n_seg_classes):
        mean_dice = np.mean(dices[:, idx])
        print('Mean Dice for class-{} is {:.4f}'.format(idx, mean_dice))
