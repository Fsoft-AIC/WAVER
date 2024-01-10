from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset


class MSRVTTDataset(RetrievalDataset):
    """MSRVTT dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, top_k=1, vocab_type="activity", mode='all', prompt_tuning=True, config=None):
        super(MSRVTTDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, prompt_tuning, config=config)
        pass

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """
        csv_path = {'train': join(self.anno_path, 'MSRVTT_train.9k.csv'),
                    'val': join(self.anno_path, 'MSRVTT_JSFUSION_test.csv'),
                    'test': join(self.anno_path, 'MSRVTT_JSFUSION_test.csv')}[subset]
        if exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError
        video_id_list = list(csv['video_id'].values)
        vocab_dict = OrderedDict()
        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        sentences_per_video_dict = OrderedDict()
        local_anno_path = "/home/synh/workspace/huy/video_retrieval/oc-t2v-retrieval/data/MSR-VTT/matching_json"
        if subset == 'train':
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            vocab_path = join(local_anno_path, 'msrvtt_train_12-frames_MSRVTT_activity_verb_vocab_50482_clean_frequency_1.json')
            vocab_dict = json.load(open(vocab_path, 'r'))
            data = json.load(open(anno_path, 'r'))
            for itm in data['annotations']:
                if itm['image_id'] in video_id_list:
                    sentences_dict[len(sentences_dict)] = (itm['image_id'], (itm['caption'], None, None))
                    # if itm['image_id'] not in video_dict:
                    video_dict[itm['image_id']] = join(self.video_path, "{}.mp4".format(itm['image_id']))
                    if itm['image_id'] not in sentences_per_video_dict:
                        sentences_per_video_dict[itm['image_id']] = itm['caption']
        else:
            vocab_path = join(local_anno_path, 'msrvtt_test_12-frames_MSRVTT_activity_verb_vocab_50482_clean_frequency_1.json')
            vocab_dict = json.load(open(vocab_path, 'r'))
            for _, itm in csv.iterrows():
                sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['sentence'], None, None))
                # if itm['video_id'] not in video_dict:
                video_dict[itm['video_id']] = join(self.video_path, "{}.mp4".format(itm['video_id']))
                if itm['video_id'] not in sentences_per_video_dict:
                    sentences_per_video_dict[itm['video_id']] = itm['sentence']

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict, sentences_per_video_dict, vocab_dict
