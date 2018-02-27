from __future__ import division
import numpy as np
import os
import cv2

class JumpData:
    def __init__(self):
        self.data_dir = '/Users/yxue/Documents/hackthon1/drive-download-20180118T221110Z-001/data'
        self.name_list = []
        self.get_name_list()
        self.val_name_list = self.name_list[:200]
        self.train_name_list = self.name_list[200:]

    def get_name_list(self):
        for i in range(3, 10):
            dir = os.path.join(self.data_dir, 'exp_%02d' % i)
            this_name = os.listdir(dir)
            this_name = [os.path.join(dir, name) for name in this_name]
            self.name_list = self.name_list + this_name
        self.name_list_raw = self.name_list
        self.name_list = filter(lambda name: 'res' in name, self.name_list)
        self.name_list = list(self.name_list)

        def _name_checker(name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            if img_name in self.name_list_raw:
                return True
            else:
                return False

        self.name_list = list(filter(_name_checker, self.name_list))

    def next_batch(self, batch_size=8):
        batch_name = np.random.choice(self.train_name_list, batch_size)
        batch = {}
        for idx, name in enumerate(batch_name):
            print name
            position_index = name.index('_res')
            # state_313.png
            img_name = name[:position_index] + '.png'
            # state_000_res_h_608_w_446.png
            # x is h
            # y is w
            x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
            # get white central
            x, y = int(x), int(y)
            # read state_313.png
            img = cv2.imread(img_name)
            # 720(w) * 640(h)
            img = img[320: -320, :, :]
            # resize img to 72(w) * 64(h) * 3(rgb)
            scale = 5
            img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
            print img.shape
            # 128(h) 144(w) 3(rgb)
            # resize label
            x = x - 320
            x = x / float(scale)
            y = y / float(scale)
            print 'x = %2d, y = %2d' % (x,y)
            # print 'writing img...'
            # cv2.imwrite(img_name + '.png',img)
            label = np.array([x, y], dtype=np.float32)
            # mask1 = (img[:, :, 0] == 245)
            # mask2 = (img[:, :, 1] == 245)
            # mask3 = (img[:, :, 2] == 245)
            # mask = mask1 * mask2 * mask3
            # img[mask] = img[x - 320 + 10, y + 14, :]
            if idx == 0:
                # 72(w) * 64(h) * 3(rgb)
                batch['img'] = img[np.newaxis, :, :, :]
                batch['label'] = label.reshape([1, label.shape[0]])
            else:
                # 72(w) * 64(h) * 3(rgb)
                img_tmp = img[np.newaxis, :, :, :]
                label_tmp = label.reshape((1, label.shape[0]))
                batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
        return batch


if __name__ == '__main__':
    data = JumpData()
    batch = data.next_batch(1)
    print(batch['img'].shape)
