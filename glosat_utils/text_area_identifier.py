import numpy as np
from PIL import Image

from utils.image import resize

from utils.constant import LABEL_TO_COLOR_MAPPING
from utils.image import LabeledArray2Image
import torch

from models import load_model_from_path
from utils import coerce_to_path_and_check_exist
from utils.path import MODELS_PATH
from utils.constant import MODEL_FILE
from PIL import Image
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

TAG = 'default'
model_path = coerce_to_path_and_check_exist(MODELS_PATH / TAG / MODEL_FILE)
model_extractor, (img_size, restricted_labels, normalize) = load_model_from_path(model_path, device=device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
_ = model_extractor.eval()

def find_text_region(img, label_idx_color_mapping, normalize):

    im_pil = Image.fromarray(img)
    
    # Normalize and convert to Tensor
    inp = np.array(img, dtype=np.float32) / 255
    if normalize:
        inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(device)

    # compute prediction
    pred = model_extractor(inp.reshape(1, *inp.shape))[0].max(0)[1].cpu().numpy()

    # Retrieve good color mapping and transform to image
    pred_img = LabeledArray2Image.convert(pred, label_idx_color_mapping)

    # Blend predictions with original image    
    mask = Image.fromarray((np.array(pred_img) == (0, 0, 0)).all(axis=-1).astype(np.uint8) * 127 + 128)
    blend_img = Image.composite(im_pil, pred_img, mask)
    
    mask = torch.from_numpy(pred)
    mask = mask.unsqueeze(0)

    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask == obj_ids[:, None, None]

    return masks, blend_img 

