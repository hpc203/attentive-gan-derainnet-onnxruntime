import cv2
import numpy as np
import onnxruntime


class Derainnet():
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)
        self.input_height, self.input_width = self.net.get_inputs()[0].shape[2:]
        self.input_name = self.net.get_inputs()[0].name

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = img.astype(np.float32) / 127.5 - 1.0
        blob = img.transpose(2, 0, 1)[np.newaxis, ...]
        outs = self.net.run(None, {self.input_name: blob})
        
        output_image = outs[0].squeeze()
        for i in range(output_image.shape[2]):
            min_val = np.min(output_image[:, :, i])
            max_val = np.max(output_image[:, :, i])
            output_image[:, :, i] = (output_image[:, :, i] - min_val) * 255.0 / (max_val-min_val)
        output_image = output_image.astype(np.uint8)
        output_image = cv2.resize(output_image, (srcimg.shape[1], srcimg.shape[0]))
        return output_image
    
if __name__=='__main__':
    imgpath = 'sample.png'
    modelpath = 'weights/attentive_gan_derainnet_240x360.onnx'
    
    mynet = Derainnet(modelpath)
    srcimg = cv2.imread(imgpath)
    dstimg = mynet.detect(srcimg)

    cv2.imwrite('result.jpg', np.hstack((srcimg, dstimg)))
    # winName = 'Deep learning use onnxruntime'
    # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    # cv2.imshow(winName, np.hstack((srcimg, dstimg)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()