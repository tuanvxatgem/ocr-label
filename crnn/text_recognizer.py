import inspect
import os
import sys

import torch
import torch.nn.functional as F
import torch.utils.data

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, currentdir)

from config_web import opt

from dataset import AlignCollate, ListDataset
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter


class TextRecognizer:
    def __init__(self):
        if opt.device == 'cpu':
            self.device = torch.device('cpu')
            self.num_gpu = 0
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)

        opt.num_class = len(self.converter.character)
        if opt.rgb:
            opt.input_channel = 3

        self.model = Model(opt)

        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        dir_name = os.path.dirname(os.path.realpath(__file__))
        self.model.load_state_dict(torch.load(f"{dir_name}/{opt.saved_model}", map_location=self.device))

    def internal_inference(self, list_img):
        result = []
        align_collate_demo = AlignCollate(img_h=opt.imgH, img_w=opt.imgW, keep_ratio_with_pad=opt.PAD,
                                          vertical_lettering=opt.vertical_lettering)
        data = ListDataset(list_img=list_img, opt=opt)  # use RawDataset
        loader = torch.utils.data.DataLoader(
            data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=align_collate_demo)

        self.model.eval()
        with torch.no_grad():
            for i, (image_tensors, str_index) in enumerate(loader):
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)

                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in opt.Prediction:
                    predictions = self.model(image, text_for_pred)

                    # Select max probability (greedy decoding) then decode index to character
                    predictions_size = torch.IntTensor([predictions.size(1)] * batch_size)
                    _, predictions_index = predictions.max(2)
                    predictions_str = self.converter.decode(predictions_index, predictions_size)

                else:
                    predictions = self.model(image, text_for_pred, is_train=False)

                    # select max probability (greedy decoding) then decode index to character
                    _, predictions_index = predictions.max(2)
                    predictions_str = self.converter.decode(predictions_index, length_for_pred)

                predictions_prob = F.softmax(predictions, dim=2)
                predictions_max_prob, _ = predictions_prob.max(dim=2)
                for pred, pred_max_prob in zip(predictions_str, predictions_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_eos = pred.find('[s]')
                        pred = pred[:pred_eos]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_eos]
                    try:
                        # calculate confidence score (= multiply of pred_max_prob)
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                        result.append(pred)
                    except Exception as e:
                        print(f"Error! so pass {e}")
                        print(pred_max_prob)
                        result.append(pred)
        return result
