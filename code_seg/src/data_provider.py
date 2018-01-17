import numpy as np
import cv2
import sys
import cfg
import os
#import skimage.io
from pycocotools.coco import COCO
#import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
from matplotlib.patches import Polygon
import pycocotools.mask as maskUtils


class DataProvider:
    def __len__(self):
        assert False

class COCO_detection_train(DataProvider):
    def __init__(self, root_dir, json_file):
        self.root = root_dir
        self.json_file = json_file
        self.img_root = osp.join(self.root, 'images')
        annFile = osp.join(self.root, 'annotations', self.json_file)
        self.coco = COCO(annFile)
        self.imgIds = sorted(set([a['image_id'] for a in self.coco.anns.values()]))
        self.cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(self.cat_ids)
        self.cat_names = [str(cat['name']) for cat in cats]
        self.rewind()


    def rewind(self):
        np.random.shuffle(self.imgIds)
    def __len__(self):
        return len(self.imgIds)


    def gt_to_label(self, gt):
        gt_tmp = gt.flatten().tolist()
        label = map(lambda x: list(self.cat_ids).index(x) if x > 0 else -1, gt_tmp)
        return np.array(label).reshape(gt.shape)


    def label_to_catid(self, label):
        label_tmp = label.flatten().tolist()
        gt = map(lambda x: list(self.cat_ids)[int(x)] if x >= 0 else 0, label_tmp)
        return np.array(gt).reshape(label.shape)


    def decode_ann(self, anns, im_h, im_w):
        boxes = []
        areas = []
        polys = []
        cat_ids = []
        mask_label = np.zeros((im_h, im_w), dtype=np.int32)
        m = np.zeros((im_h, im_w), dtype=np.int32)

        for ann in anns:
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    for seg in ann['segmentation']:
                        cat_id = ann['category_id']
                        areas.append(ann['area'])
                        boxes.append(ann['bbox'])
                        poly = np.array(seg).reshape((-1, int(len(seg) / 2), 2))
                        polys.append(poly)
                        cat_ids.append(cat_id)
                        #cv2.fillPoly(mask_label, [poly_org], (cat_id, cat_id, cat_id))
                else:

                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], im_h, im_w)
                    else:
                        rle = [ann['segmentation']]
                    m += maskUtils.decode(rle)[:,:,0]

        ord_area = np.argsort(np.array(areas))[::-1]
        instance_num = len(boxes)
        boxes = np.array(boxes).reshape(-1,4)
        boxes = boxes[ord_area]
        for n in range(instance_num):
            poly = polys[ord_area[n]]
            c = cat_ids[ord_area[n]]
            poly = np.ascontiguousarray(poly, np.int)
            cv2.fillPoly(mask_label, [poly], (c, c, c))

        return boxes, mask_label*(1.0-m)


    def __getitem__(self, item):
        sel_id = self.imgIds[item]
        imgName = self.coco.loadImgs(ids=sel_id)[0]['file_name'].replace('.jpg', '')
        image = cv2.imread(osp.join(self.img_root, imgName + '.jpg'))
        #print 'item: {}, imname:{}'.format(item, osp.join(self.img_root, imgName + '.jpg'))
        annId = self.coco.getAnnIds(imgIds=sel_id)
        anns = self.coco.loadAnns(annId)
        im_h, im_w = image.shape[:2]
        boxes, gt = self.decode_ann(anns, im_h, im_w)
        label_gt = self.gt_to_label(gt)
        # print 'gt: ', np.unique(gt)
        # print 'label_gt: ', np.unique(label_gt)
        # print 're gt: ', np.unique(self.label_to_catid(label_gt))

        assert np.max(label_gt) < len(self.cat_ids)
        return image, label_gt, boxes, anns


class MultiDataProvider(DataProvider):
    def __init__(self, providers, crop_size, min_ratio, max_ratio, hor_flip):
        assert len(providers) > 0
        self.providers = providers
        self.crop_size = crop_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.inds = np.arange(len(self))
        self.cur_ind = -1
        self.hor_flip = hor_flip
        self.rewind()

    def rewind(self):
        self.cur_index = -1
        np.random.shuffle(self.inds)

    def __len__(self):
        length = 0
        for provider in self.providers:
            length += len(provider)
        return length

    def get_provider_id(self, ind):
        for provider_ind, provider in enumerate(self.providers):
            if ind < len(provider):
                return provider_ind, ind
            else:
                ind -= len(provider)
        raise IndexError

    def get_next(self):
        if self.cur_ind >= len(self.inds) - 1:
            print 'rewind'
            self.rewind()
        self.cur_ind += 1

        if self.cur_ind >= len(self):
            self.cur_ind -= len(self)

        idx = self.inds[self.cur_ind]
        # idx = 0
        self.cur_ind += 1

        assert idx < len(self)
        assert idx >= 0

        provider_index, index = self.get_provider_id(idx)


        image, label, boxes, _ = self.providers[provider_index][index]
        im_h, im_w = image.shape[:2]
        #print('minratio: {}, max_ratio: {}'.format(int(self.min_ratio*100), int(self.max_ratio*100)))
        ratio = np.random.randint(int(self.min_ratio*100), int(self.max_ratio*100), 1) / 100.0
        #ratio = 1
        #print 'ratio:', ratio
        dst_h, dst_w = int(im_h*ratio), int(im_w*ratio)
        image = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
        #boxes = boxes * ratio
        padded_img = image - 122.0
        padded_label = label
        im_h, im_w = padded_img.shape[:2]

        #print 'im_h: {}, im_w: {}'.format(im_h, im_w)

        if im_h < self.crop_size:
            pad = np.zeros((self.crop_size-im_h, padded_img.shape[1], padded_img.shape[2]))
            padded_img = np.concatenate((padded_img, pad), axis=0)

            pad = np.zeros((self.crop_size - im_h, padded_img.shape[1])) - 1
            padded_label = np.concatenate((padded_label, pad), axis=0)

        if im_w < self.crop_size:
            pad = np.zeros((padded_img.shape[0], self.crop_size-im_w, padded_img.shape[2]))
            padded_img = np.concatenate((padded_img, pad), axis=1)

            pad = np.zeros((padded_img.shape[0], self.crop_size - im_w))-1
            padded_label = np.concatenate((padded_label, pad), axis=1)

        assert padded_img.shape[0] >= self.crop_size, \
            'padded image size should be greater than crop size: {} >= {}'.format(padded_img.shape[0], self.crop_size)
        assert padded_img.shape[1] >= self.crop_size, \
            'padded image size should be greater than crop size: {} >= {}'.format(padded_img.shape[1], self.crop_size)
        assert padded_label.shape[0] >= self.crop_size, \
            'padded label size should be greater than crop size: {} >= {}'.format(padded_label.shape[0], self.crop_size)
        assert padded_label.shape[1] >= self.crop_size, \
            'padded label size should be greater than crop size: {} >= {}'.format(padded_label.shape[1], self.crop_size)

        if self.crop_size - padded_img.shape[1] > 0:
            sx = int(np.random.randint((0, self.crop_size-padded_img.shape[1])))
        else:
            sx = 0

        if self.crop_size - padded_img.shape[0] > 0:
            sy = int(np.random.randing((0, self.crop_size-padded_img.shape[0])))
        else:
            sy = 0
        # sx=0
        # sy=0
        crop_image = np.float32(padded_img[sy:sy+self.crop_size, sx:sx+self.crop_size, :])
        crop_label = np.float32(padded_label[sy:sy+self.crop_size, sx:sx+self.crop_size])

        crop_image_flip = np.float32(crop_image)
        crop_label_flip = np.float32(crop_label)

        if self.hor_flip:
            crop_image_flip = crop_image[:, ::-1, :]
            crop_label_flip = crop_label[:, ::-1]

        tmp = np.random.rand(1)
        if tmp > 0.5:
            return crop_image, crop_label
        else:
            return crop_image_flip, crop_label_flip

    def get_batch(self, batch_size):
        res_imgs = np.zeros((batch_size, self.crop_size, self.crop_size, 3), dtype=np.float32)
        res_labels = np.zeros((batch_size, self.crop_size, self.crop_size), dtype=np.float32)

        for n in range(batch_size):
            img, label = self.get_next()
            res_imgs[n] = img
            res_labels[n, :, :] = label

        return res_imgs, res_labels


def test_coco_detection_train():
    import matplotlib.pyplot as plt
    root = '/home/tonghe/Downloads/coco'
    print 'len train:', len(os.listdir(osp.join(root, 'images')))
    json_file = 'instances_train2017.json'
    coco = COCO_detection_train(root, json_file)
    print 'len: ', len(coco)
    print coco.cat_names
    coco2 = coco.coco
    for n in xrange(0, len(coco)):
        print '%d / %d' % (n, len(coco))
        image, label, boxes, anns = coco[n]
        plt.subplot(131)
        plt.imshow(np.uint8(image[:,:,::-1]))
        plt.subplot(132)
        plt.imshow(label)

        plt.subplot(133)
        plt.imshow(np.uint8(image[:,:,::-1]))
        coco2.showAnns(anns)
        plt.show()

if __name__ == '__main__':
    test_coco_detection_train()
