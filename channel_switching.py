from PIL import Image
import os

path = "E:\\Pix2Pix\\data\\clothes\\val"
filelist = os.listdir(path)


# 数据集处理（有些ps生成的图片为四通道，需要转换为三通道）

def main():
    for file in filelist:
        whole_path = os.path.join(path, file)
        img = Image.open(whole_path)
        img = img.convert("RGB")
        save_path = 'E:\\Pix2Pix\\data\\clothes\\newval\\'
        img.save(save_path + file)


if __name__ == "__main__":
    main()
