import utils
device = "cpu"
saved_model = 'best_accuracy.pth'
image_folder = 'val_images_20220318'
ground_truth = "ground_truth_val_20220318.txt"
log_test = f"log/test/test_{saved_model[(saved_model.rfind('/') + 1):].replace(':', '')}_{image_folder.replace('/', '_')}.xlsx"
workers = 4
batch_size = 2
batch_max_length = 100
imgH = 32
imgW = 100
rgb = False
vertical_lettering = False
character_file = "japanese_characters.txt"
character = utils.read_character(character_file)
# print(len(character))
PAD = True
Transformation = 'None'
FeatureExtraction = 'ResNet'
SequenceModeling = 'BiLSTM'
Prediction = 'Attn'
num_fiducial = 20
input_channel = 1
hidden_size = 256
output_channel = 512

dict_opt = {
    'device': device,
    'ground_truth': ground_truth,
    'log_test': log_test,
    'image_folder': image_folder,
    'workers': workers,
    'batch_size': batch_size,
    'saved_model': saved_model,
    'batch_max_length': batch_max_length,
    'imgH': imgH,
    'imgW': imgW,
    'rgb': rgb,
    'character': character,
    'PAD': PAD,
    'Transformation': Transformation,
    'FeatureExtraction': FeatureExtraction,
    'SequenceModeling': SequenceModeling,
    'Prediction': Prediction,
    'num_fiducial': num_fiducial,
    'input_channel': input_channel,
    'output_channel': output_channel,
    'hidden_size': hidden_size,
    'vertical_lettering': vertical_lettering,
}


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opt = DotDict(dict_opt)
